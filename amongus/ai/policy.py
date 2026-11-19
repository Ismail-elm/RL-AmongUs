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


def sample(rng, pr):
    """Même tirage que rng.choice(len(pr), p=pr) (même nombre aléatoire consommé, même résultat), sans ses
    vérifications coûteuses."""
    cdf = np.asarray(pr, dtype=np.float64).cumsum()
    cdf /= cdf[-1]
    return int(cdf.searchsorted(rng.random(), side="right"))


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
        a = sample(self.rng, pr)
        if not record:
            return a, None
        logp = float(np.log(max(pr[a], 1e-12)))
        if self.temperature != 1.0:            # on n'enregistre que des échantillons « on-policy »
            raise ValueError("enregistrement impossible avec temperature != 1")
        return a, dict(obs=o, crit=crit, mask=mask, act=a, logp=logp)


# ------------------------------------------------------------------ fichiers
def save_actors(path, crew_params, imp_params, meta=None, crew_vote=None, imp_vote=None, imp_claim=None):
    """Sauve les acteurs (poids numpy) dans un .npz + meta JSON. Les réseaux de vote et de déclaration sont optionnels."""
    d = {}
    for role, params in (("crew", crew_params), ("imp", imp_params), ("crewvote", crew_vote), ("impvote", imp_vote),
                         ("impclaim", imp_claim)):
        for i, (w, b) in enumerate(params or []):
            d[f"{role}_W{i}"] = np.asarray(w, np.float32)
            d[f"{role}_b{i}"] = np.asarray(b, np.float32)
    np.savez(path, **d)
    if meta is not None:
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def load_actors(path):
    """-> (NumpyActor crew, NumpyActor imposteur). Refuse un modèle entraîné avec d'autres observations."""
    z = np.load(path)
    expected = O.dims()[:2]
    out = []
    for role, dim in zip(("crew", "imp"), expected):
        n = sum(1 for k in z.files if k.startswith(role + "_W"))
        got = z[f"{role}_W0"].shape[0]
        if got != dim:
            raise ValueError(f"{path} : modèle d'une autre version du jeu (observation {role} de taille {got}, "
                             f"{dim} attendue). Réentraînez avec train_mappo.py ou prenez un modèle de cette version.")
        out.append(NumpyActor([(z[f"{role}_W{i}"], z[f"{role}_b{i}"]) for i in range(n)]))
    return out[0], out[1]


def load_claim_actor(path):
    """-> NumpyActor de déclaration de l'imposteur, ou None (modèle sans, ou d'une autre version)."""
    from .claim import dims
    z = np.load(path)
    n = sum(1 for k in z.files if k.startswith("impclaim_W"))
    if not n or z["impclaim_W0"].shape[0] != dims()[0]:
        return None
    return NumpyActor([(z[f"impclaim_W{i}"], z[f"impclaim_b{i}"]) for i in range(n)])


def load_vote_actors(path):
    """-> (NumpyActor de vote crew | None, NumpyActor de vote imposteur | None) ; None si le modèle n'en a pas."""
    from .vote import dims
    z = np.load(path)
    out = []
    for role in ("crewvote", "impvote"):
        n = sum(1 for k in z.files if k.startswith(role + "_W"))
        if n and z[f"{role}_W0"].shape[0] != dims()[0]:
            n = 0        # réseau de vote d'une ancienne version (autre observation) : on retombe sur la règle heuristique
        out.append(NumpyActor([(z[f"{role}_W{i}"], z[f"{role}_b{i}"]) for i in range(n)]) if n else None)
    return out[0], out[1]
