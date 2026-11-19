"""Collecte de parties en parallèle (sans PyTorch). Utilisé par train_mappo.py et l'évaluation."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..rt_world import RTConfig, RTWorld, RewardConfig, run_game
from .bots import RandomCtrl, ScriptedCrew, ScriptedImposter
from .policy import NumpyActor, PolicyCtrl
from .claim import ClaimCtrl
from .vote import TeacherVote, VoteCtrl

_world = None


def _get_world():
    global _world
    if _world is None:
        _world = RTWorld(Config.players(), RTConfig(), seed=0, rw=RewardConfig())
    return _world


def make_ctrl(src, role, seed, greedy=False):
    """src : ('net', params) | ('script', nom)."""
    kind, val = src
    if kind == "net":
        return PolicyCtrl(NumpyActor(val), seed=seed, greedy=greedy)
    if role == "crew":
        return {"scripted": ScriptedCrew, "random": lambda: RandomCtrl(seed)}[val]()
    return {"scripted": lambda: ScriptedImposter(seed), "prudent": lambda: ScriptedImposter(seed, prudent=True),
            "random": lambda: RandomCtrl(seed)}[val]()


def make_vote(src, seed, greedy=False):
    """src : None | ('script', ...) -> règle heuristique ; ('net', params) -> vote appris ;
    ('teacher', None) -> règle heuristique + exemples étiquetés pour le pré-entraînement."""
    if src is not None and src[0] == "teacher":
        return TeacherVote()
    if src is None or src[0] != "net" or src[1] is None:
        return None
    return VoteCtrl(NumpyActor(src[1]), seed=seed, greedy=greedy)


def make_claim(src, seed, greedy=False):
    """src : None -> règle heuristique ; ('net', params) -> déclaration apprise de l'imposteur."""
    if src is None or src[0] != "net" or src[1] is None:
        return None
    return ClaimCtrl(NumpyActor(src[1]), seed=seed, greedy=greedy)


def _stack(traj):
    return dict(
        obs=np.stack([s["obs"] for s in traj]), crit=np.stack([s["crit"] for s in traj]),
        mask=np.stack([s["mask"] for s in traj]), act=np.array([s["act"] for s in traj], dtype=np.int64),
        logp=np.array([s["logp"] for s in traj], dtype=np.float32),
        r=np.array([s["r"] for s in traj], dtype=np.float32), dur=np.array([s["dur"] for s in traj], dtype=np.float32),
        done=np.array([s["done"] for s in traj], dtype=np.float32))


def play_batch(task):
    """task : dict(specs=[dict(seed, crew, imp, crew_vote, imp_vote, imp_claim, rec_crew, rec_imp)], greedy=bool).
    Renvoie (stats par partie, trajectoires crew, trajectoires imposteur, votes crew, votes imposteur,
    déclarations de l'imposteur)."""
    w = _get_world()
    stats, crew_tr, imp_tr, crew_vt, imp_vt, imp_ct = [], [], [], [], [], []
    for sp in task["specs"]:
        w.rng.seed(sp["seed"])
        crew = make_ctrl(sp["crew"], "crew", sp["seed"] + 1, task.get("greedy", False))
        imp = make_ctrl(sp["imp"], "imp", sp["seed"] + 2, task.get("greedy", False))
        greedy = task.get("greedy", False)
        r = run_game(w, crew, imp, record_crew=sp.get("rec_crew", False), record_imp=sp.get("rec_imp", False),
                     crew_vote=make_vote(sp.get("crew_vote"), sp["seed"] + 3, greedy),
                     imp_vote=make_vote(sp.get("imp_vote"), sp["seed"] + 4, greedy),
                     imp_claim=make_claim(sp.get("imp_claim"), sp["seed"] + 5, greedy))
        r["tag"] = sp.get("tag")
        stats.append(r)
        for q in range(w.n):
            if q in w.imps:
                if sp.get("rec_imp"):
                    for src, dst in ((w.traj, imp_tr), (w.vtraj, imp_vt), (w.ctraj, imp_ct)):
                        if src[q]:
                            dst.append(_stack(src[q]))
            elif sp.get("rec_crew"):
                for src, dst in ((w.traj, crew_tr), (w.vtraj, crew_vt)):
                    if src[q]:
                        dst.append(_stack(src[q]))
    return stats, crew_tr, imp_tr, crew_vt, imp_vt, imp_ct
