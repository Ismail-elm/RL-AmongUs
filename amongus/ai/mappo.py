"""MAPPO (PyTorch) : acteurs décentralisés par rôle, critiques centralisés, PPO + GAE semi-markovien.

* Un acteur PARTAGÉ par tous les crewmates (paramètres partagés) et un acteur pour l'imposteur.
* Un critique par rôle, qui voit l'état global (CTDE : entraînement centralisé, exécution décentralisée).
* Décisions asynchrones (macro-actions de durée variable) : GAE avec escompte gamma**(durée/tau).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env import N_ACTIONS


def _init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(i, h, o, out_gain):
    return nn.Sequential(_init(nn.Linear(i, h), 2 ** 0.5), nn.Tanh(), _init(nn.Linear(h, h), 2 ** 0.5), nn.Tanh(),
                         _init(nn.Linear(h, o), out_gain))


class RoleLearner:
    def __init__(self, obs_dim, crit_dim, hidden=128, lr=3e-4, crit_hidden=256, device="cpu", n_actions=N_ACTIONS):
        self.dev = torch.device(device)
        self.actor = mlp(obs_dim, hidden, n_actions, 0.01).to(self.dev)
        self.critic = mlp(crit_dim, crit_hidden, 1, 1.0).to(self.dev)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=1e-5)
        self.v_mean, self.v_var = 0.0, 1.0

    def actor_params(self):
        layers = [m for m in self.actor if isinstance(m, nn.Linear)]
        return [(l.weight.detach().cpu().numpy().T.copy(), l.bias.detach().cpu().numpy().copy()) for l in layers]

    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt=self.opt.state_dict(),
                    v=(self.v_mean, self.v_var))

    def load(self, st):
        actor = dict(st["actor"])
        last = max(int(k.split(".")[0]) for k in actor)
        w, b = actor[f"{last}.weight"], actor[f"{last}.bias"]
        n_out = self.actor[last].out_features
        grown = w.shape[0] < n_out
        if grown:
            # modèle entraîné avec moins d'actions (avant « signaler ») : on ajoute les sorties manquantes, peu probables
            # au départ mais explorables, sans toucher au reste du réseau.
            k = n_out - w.shape[0]
            actor[f"{last}.weight"] = torch.cat([w, torch.zeros(k, w.shape[1], dtype=w.dtype, device=w.device)])
            actor[f"{last}.bias"] = torch.cat([b, torch.full((k,), float(b.mean()) - 2.0, dtype=b.dtype, device=b.device)])
        self.actor.load_state_dict(actor)
        self.critic.load_state_dict(st["critic"])
        if not grown:                  # l'état d'Adam n'a plus la bonne forme si le réseau a grandi
            self.opt.load_state_dict(st["opt"])
        self.v_mean, self.v_var = st["v"]

    @torch.no_grad()
    def values(self, crit):
        v = self.critic(torch.as_tensor(crit).to(self.dev)).squeeze(-1).cpu().numpy()
        return v * np.sqrt(self.v_var + 1e-8) + self.v_mean

    def imitate(self, trajs, epochs=30, mb=256, lr=1e-3):
        """Pré-entraînement supervisé de l'acteur : entropie croisée entre ses logits (masqués) et l'action `act`.
        Renvoie la précision finale sur ces exemples."""
        cat = lambda k: torch.as_tensor(np.concatenate([t[k] for t in trajs])).to(self.dev)
        obs, mask, act = cat("obs"), cat("mask"), cat("act")
        opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        N = len(act)
        for _ in range(epochs):
            perm = torch.randperm(N, device=self.dev)
            for s in range(0, N, mb):
                ix = perm[s:s + mb]
                loss = nn.functional.cross_entropy(self.actor(obs[ix]).masked_fill(~mask[ix], -1e9), act[ix])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = self.actor(obs).masked_fill(~mask, -1e9).argmax(-1)
        return float((pred == act).float().mean()), N

    def update(self, trajs, gamma=0.98, tau=3.5, lam=0.95, clip=0.2, epochs=4, mb=2048, ent=0.01, vf=0.5, lr=None,
               max_kl=0.03):
        if not trajs:
            return {}
        if lr is not None:
            for g in self.opt.param_groups:
                g["lr"] = lr
        # --- GAE par trajectoire
        advs, rets = [], []
        for t in trajs:
            v = self.values(t["crit"])
            T = len(v)
            g = gamma ** (t["dur"] / tau)
            adv = np.zeros(T, dtype=np.float32)
            last = 0.0
            for i in range(T - 1, -1, -1):
                nv = 0.0 if (t["done"][i] or i == T - 1) else v[i + 1]
                nonterm = 0.0 if (t["done"][i] or i == T - 1) else 1.0
                delta = t["r"][i] + g[i] * nv - v[i]
                last = delta + g[i] * lam * nonterm * last
                adv[i] = last
            advs.append(adv)
            rets.append(adv + v)
        cat = lambda k: torch.as_tensor(np.concatenate([t[k] for t in trajs])).to(self.dev)
        obs, crit, mask, act, logp0 = cat("obs"), cat("crit"), cat("mask"), cat("act"), cat("logp")
        adv = torch.as_tensor(np.concatenate(advs)).to(self.dev)
        ret = np.concatenate(rets)
        # --- normalisation des retours (valeur du critique en unités normalisées)
        self.v_mean = 0.9 * self.v_mean + 0.1 * float(ret.mean())
        self.v_var = 0.9 * self.v_var + 0.1 * float(ret.var())
        retn = torch.as_tensor(((ret - self.v_mean) / np.sqrt(self.v_var + 1e-8)).astype(np.float32)).to(self.dev)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = len(act)
        stats = dict(n=N, pl=0.0, vl=0.0, ent=0.0, kl=0.0, clipfrac=0.0)
        cnt = 0
        for ep in range(epochs):
            perm = torch.randperm(N, device=self.dev)
            stop = False
            for s in range(0, N, mb):
                ix = perm[s:s + mb]
                lg = self.actor(obs[ix]).masked_fill(~mask[ix], -1e9)
                dist = torch.distributions.Categorical(logits=lg)
                lp = dist.log_prob(act[ix])
                ratio = torch.exp(lp - logp0[ix])
                a = adv_n[ix]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                v = self.critic(crit[ix]).squeeze(-1)
                vl = nn.functional.smooth_l1_loss(v, retn[ix])
                e = dist.entropy().mean()
                loss = pl + vf * vl - ent * e
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt.step()
                with torch.no_grad():
                    kl = (logp0[ix] - lp).mean().item()
                    stats["pl"] += pl.item(); stats["vl"] += vl.item(); stats["ent"] += e.item()
                    stats["kl"] += kl
                    stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
                    cnt += 1
                if kl > max_kl:
                    stop = True
                    break
            if stop:
                break
        for k in ("pl", "vl", "ent", "kl", "clipfrac"):
            stats[k] /= max(cnt, 1)
        return stats
