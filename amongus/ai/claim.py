"""Déclaration APPRISE de l'imposteur en réunion : où il dit avoir été au moment du meurtre (vrai ou faux), et qui il
accuse (« je l'ai vu tuer ») — ou personne.

Il parle après les crewmates (qui disent la vérité) et sait donc ce qu'ils ont déclaré. Il décide à partir :
* de la salle du crime, de ce que chaque crewmate a déclaré (salle par salle) ;
* de ce qu'il a vu lui-même au moment du meurtre (qui il voyait, donc qui a pu le voir, et dans quelle salle) ;
* de ce qui pèse déjà contre lui (accusations des témoins, « vu avec la victime », a-t-il signalé le corps).
Action : salle (12) × accusation (6 autres joueurs en ordre relatif, ou personne) = 84 choix.
Il apprend comme le vote : chaque déclaration est un épisode d'un pas, récompensé par le résultat de la réunion.
"""
from __future__ import annotations

import numpy as np

from . import obs as O
from . import vote as V
from ..mapdata import DIST, N_ROOMS

N_ACC = O.MAX_PLAYERS           # jusqu'à 9 autres joueurs + personne
NO_ACC = N_ACC - 1
N_CLAIM = N_ROOMS * N_ACC


def claim_obs(world, imp, ctx) -> np.ndarray:
    n = world.n
    kr = ctx.kill_room
    ref = ctx.ref
    sight = ref.sight.get(imp, set()) if ref is not None else set()
    maxd = max(max(r) for r in DIST)
    claimed = {}
    for q, r in ctx.claims.items():
        if q != imp:
            claimed[r] = claimed.get(r, 0) + 1
    seen_in = {}
    for q in sight:
        z = ref.zone.get(q, -1)
        seen_in[z] = seen_in.get(z, 0) + 1
    against = sum(1 for a, t in ctx.accuse.items() if t == imp)
    pub = getattr(ctx, "pub", None)
    if pub is None:
        pub = ctx.pub = V.public_testimony(world, ctx)
    vpub, _, _ = V.victim_testimony(world, ctx, imp)
    g = np.array([len(world.alive_players()) / O.MAX_PLAYERS, 1.0 if ctx.emergency else 0.0, 1.0 if ctx.reporter == imp else 0.0,
                  min(len(sight), 3) / 3.0, min(against, 3) / 3.0,
                  1.0 if vpub[imp] is not None else 0.0, vpub[imp][1] if vpub[imp] else 0.0,
                  1.0 if pub[imp] is not None else 0.0, min(len(ctx.bodies), 3) / 3.0], dtype=np.float32)
    rooms = np.zeros((N_ROOMS, 4), dtype=np.float32)
    for r in range(N_ROOMS):
        rooms[r] = (min(claimed.get(r, 0), 3) / 3.0, 1.0 if r == kr else 0.0,
                    DIST[r][kr] / maxd if kr >= 0 else 0.0, min(seen_in.get(r, 0), 3) / 3.0)
    oth = np.zeros((O.MAX_PLAYERS - 1, 10), dtype=np.float32)
    for k in range(1, n):
        q = (imp + k) % n
        row = oth[k - 1]
        row[0] = 1.0 if world.alive[q] else 0.0
        row[1] = 1.0 if ctx.accuse.get(q) == imp else 0.0
        row[2] = 1.0 if ctx.claims.get(q) == kr else 0.0
        row[3] = min(len(ctx.confirm.get(q, ())), 3) / 3.0
        row[4] = 1.0 if q in sight else 0.0
        row[5] = 1.0 if q == ctx.reporter else 0.0
        row[6] = 1.0 if vpub[q] is not None else 0.0
        row[7] = 1.0 if q in world.cleared_known else 0.0
        row[8] = 1.0                                   # case occupée
        row[9] = 1.0 if q in world.imps else 0.0       # partenaire
    return np.concatenate([g, rooms.ravel(), oth.ravel()])


def claim_mask(world, imp) -> np.ndarray:
    m = np.zeros((N_ROOMS, N_ACC), dtype=bool)
    for k in range(1, world.n):
        q = (imp + k) % world.n
        m[:, k - 1] = world.alive[q] and q not in world.imps    # on n'accuse pas son partenaire
    m[:, NO_ACC] = True
    return m.ravel()


def build(world, imp, ctx):
    """-> (observation, entrée du critique, masque)."""
    o = claim_obs(world, imp, ctx)
    return o, np.concatenate([o, O.global_state(world, imp)]), claim_mask(world, imp)


def decode(world, imp, a):
    """action -> (salle, accusé | None)."""
    room, acc = divmod(int(a), N_ACC)
    return room, (None if acc == NO_ACC else (imp + acc + 1) % world.n)


def dims():
    from ..env import MeetingCtx
    from ..rt_world import RTWorld
    w = RTWorld(seed=0)
    w.reset(imposter=w.n - 1)
    ctx = MeetingCtx(reporter=0, bodies=[], cleared=set(), witnesses=[], imposter=w.n - 1)
    o, c, _ = build(w, w.n - 1, ctx)
    return len(o), len(c)


class ClaimCtrl:
    """Déclaration avec un acteur numpy. `claim` -> (salle, accusé | None, enregistrement | None)."""

    def __init__(self, actor, seed: int = 0, greedy: bool = False):
        self.actor = actor
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy

    def claim(self, world, imp, ctx, record=False):
        o, crit, mask = build(world, imp, ctx)
        lg = np.where(mask, self.actor.logits(o), -1e9)
        if self.greedy:
            a, rec = int(np.argmax(lg)), None
        else:
            z = lg - lg.max()
            pr = np.exp(z)
            pr /= pr.sum()
            a = int(self.rng.choice(N_CLAIM, p=pr))
            rec = dict(obs=o, crit=crit, mask=mask, act=a, logp=float(np.log(max(pr[a], 1e-12)))) if record else None
        room, acc = decode(world, imp, a)
        return room, acc, rec
