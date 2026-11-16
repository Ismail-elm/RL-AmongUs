"""Contrôleurs de référence (non appris) : adversaires et points de comparaison."""
from __future__ import annotations

import random

from ..env import KILL, N_ROOMS, NONE_HINT, TASK, WAIT


class ScriptedCrew:
    """Fonce sur ses tâches (aucune prudence)."""
    name = "crew scripté"

    def decide(self, world, p, legal, record=False):
        if TASK in legal:
            return TASK, None
        h = world.crew_hint(p)
        if h != NONE_HINT and h in legal:
            return h, None
        return WAIT, None


class ScriptedImposter:
    """Tue dès que possible ; sinon va là où il a vu des crewmates. `prudent` : ne tue que s'il est seul avec sa victime."""

    def __init__(self, seed=0, prudent=False):
        self.rng = random.Random(seed)
        self.prudent = prudent
        self.name = "imposteur prudent" if prudent else "imposteur scripté"

    def decide(self, world, p, legal, record=False):
        if KILL in legal:
            here = [q for q in world.crew_alive() if world.room[q] == world.room[p]]
            if not self.prudent or len(here) == 1:
                return KILL, None
        h = world.imp_hint(p)
        if h != NONE_HINT and h in legal and self.rng.random() < 0.8:
            return h, None
        moves = [a for a in legal if a < N_ROOMS]
        if moves and self.rng.random() < 0.7:
            return self.rng.choice(moves), None
        return WAIT, None


class RandomCtrl:
    name = "aléatoire"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def decide(self, world, p, legal, record=False):
        return self.rng.choice(list(legal)), None
