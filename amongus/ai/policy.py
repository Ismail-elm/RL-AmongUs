"""Politique MAPPO en numpy pur (jeu, évaluation, workers de collecte) : aucune dépendance à PyTorch."""
from __future__ import annotations

import json
import os

import numpy as np

from ..env import N_ACTIONS
from . import obs as O


class NumpyActor:
    """Perceptron tanh : obs -> logits d'actions. `params` : liste de (W, b) float32."""

    def __init__(self, params):
        self.params = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in params]

    def logits(self, x):
        h = x
        for w, b in self.params[:-1]:
            h = np.tanh(h @ w + b)
        w, b = self.params[-1]
        return h @ w + b


class PolicyCtrl:
    """Contrôleur utilisant un acteur par rôle. `decide` suit l'interface de RTWorld."""

    def __init__(self, actor: NumpyActor, seed: int = 0, greedy: bool = False, temperature: float = 1.0, name="MAPPO"):
        self.actor = actor
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy
        self.temperature = temperature
        self.name = name

    def decide(self, world, p, legal, record=False):
        o, crit, mask = O.build(world, p, legal)
        lg = self.actor.logits(o)
        if len(lg) < N_ACTIONS:        # ancien modèle (sans l'action « signaler ») : il ne la choisit jamais
            lg = np.concatenate([lg, np.full(N_ACTIONS - len(lg), -1e9, np.float32)])
        lg = np.where(mask, lg, -1e9)
        if self.greedy:
            a = int(np.argmax(lg))
            return a, None
        z = lg / self.temperature
        z = z - z.max()
        pr = np.exp(z)
        pr /= pr.sum()
        a = int(self.rng.choice(N_ACTIONS, p=pr))
        if not record:
            return a, None
        logp = float(np.log(max(pr[a], 1e-12)))
        if self.temperature != 1.0:            # on n'enregistre que des échantillons « on-policy »
            raise ValueError("enregistrement impossible avec temperature != 1")
        return a, dict(obs=o, crit=crit, mask=mask, act=a, logp=logp)


# ------------------------------------------------------------------ fichiers
def save_actors(path, crew_params, imp_params, meta=None, crew_vote=None, imp_vote=None):
    """Sauve les acteurs (poids numpy) dans un .npz + meta JSON. Les réseaux de vote sont optionnels."""
    d = {}
    for role, params in (("crew", crew_params), ("imp", imp_params), ("crewvote", crew_vote), ("impvote", imp_vote)):
        for i, (w, b) in enumerate(params or []):
            d[f"{role}_W{i}"] = np.asarray(w, np.float32)
            d[f"{role}_b{i}"] = np.asarray(b, np.float32)
    np.savez(path, **d)
    if meta is not None:
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def load_actors(path):
    """-> (NumpyActor crew, NumpyActor imposteur)."""
    z = np.load(path)
    out = []
    for role in ("crew", "imp"):
        n = sum(1 for k in z.files if k.startswith(role + "_W"))
        out.append(NumpyActor([(z[f"{role}_W{i}"], z[f"{role}_b{i}"]) for i in range(n)]))
    return out[0], out[1]


def load_vote_actors(path):
    """-> (NumpyActor de vote crew | None, NumpyActor de vote imposteur | None) ; None si le modèle n'en a pas."""
    z = np.load(path)
    out = []
    for role in ("crewvote", "impvote"):
        n = sum(1 for k in z.files if k.startswith(role + "_W"))
        out.append(NumpyActor([(z[f"{role}_W{i}"], z[f"{role}_b{i}"]) for i in range(n)]) if n else None)
    return out[0], out[1]
