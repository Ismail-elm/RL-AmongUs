"""Vote APPRIS pendant les réunions (remplace la règle heuristique `ai_vote` quand un réseau de vote est fourni).

Chaque joueur vivant choisit : voter contre l'un des 6 autres, ou passer (7 actions, masquées si la cible est morte).
Il décide à partir de :
* ce qui est PUBLIC pendant la réunion — innocentés, accusation par un témoin direct, qui a signalé le corps, et les
  témoignages mis en commun : pour chaque joueur, la dernière fois qu'un crewmate vivant l'a vu, et à quelle distance
  du corps (l'imposteur ne témoigne pas) ;
* sa MÉMOIRE privée — où et quand il a lui-même vu chaque joueur pour la dernière fois ;
* « vu avec la victime » — public et privé : la dernière fois qu'un joueur a été vu à côté de la victime, et s'il est
  le dernier à avoir été vu avec elle. C'est ce qui permet de se méfier de celui qui signale : l'imposteur peut
  signaler sa propre victime.
Le critique (entraînement seulement) voit en plus l'état global, dont l'identité de l'imposteur.
"""
from __future__ import annotations

import math

import numpy as np

from . import obs as O

N_VOTE = O.N_PLAYERS            # 6 autres joueurs (ordre relatif) + « passer »
SKIP = N_VOTE - 1
VOTE_F = 15                     # nb de features par autre joueur
N_GLOBAL = 18
AGE_PUB = 60.0                  # s : au-delà, un témoignage ne compte plus
AGE_MEM = 120.0
D_NORM = 2500.0


def _body_dist(world, x, y):
    d = min((math.hypot(bx - x, by - y) for bx, by in world.body_pos.values()), default=D_NORM)
    return min(d, D_NORM) / D_NORM


def public_testimony(world):
    """Pour chaque joueur q : (âge, distance au corps) de sa dernière observation par un AUTRE crewmate vivant."""
    n, t = world.n, world.time
    crew = world.crew_alive()
    out = []
    for q in range(n):
        best = None
        for c in crew:
            if c != q and (best is None or world.mem_t[c][q] > world.mem_t[best][q]):
                best = c
        age = t - world.mem_t[best][q] if best is not None else 1e9
        if age < AGE_PUB:
            out.append((age / AGE_PUB, _body_dist(world, world.mem_x[best][q], world.mem_y[best][q])))
        else:
            out.append(None)
    return out


def victim_testimony(world, ctx, voter):
    """Qui a été vu avec la victime ?  -> (pub, priv, frais)
    pub[q]  : (âge, est le dernier compagnon connu) d'après les crewmates vivants autres que q, ou None ;
    priv[q] : âge d'après la seule mémoire du votant, ou None ;
    frais   : âge de la dernière fois qu'une victime a été vue vivante (None si personne ne l'a vue récemment)."""
    n, t = world.n, world.time
    crew = world.crew_alive()
    victims = [v for (_r, v) in ctx.bodies]
    pub, priv = [None] * n, [None] * n
    last_alive = max((world.mem_t[c][v] for c in crew for v in victims), default=-1e9)
    for q in range(n):
        if q in victims:
            continue
        tp = max((world.pair_t[c][q][v] for c in crew if c != q for v in victims), default=-1e9)
        if t - tp < AGE_PUB:
            pub[q] = ((t - tp) / AGE_PUB, 1.0 if tp >= last_alive - 2.0 else 0.0)
        tv = max((world.pair_t[voter][q][v] for v in victims), default=-1e9) if q != voter else -1e9
        if t - tv < AGE_PUB:
            priv[q] = (t - tv) / AGE_PUB
    fresh = (t - last_alive) / AGE_PUB if t - last_alive < AGE_PUB else None
    return pub, priv, fresh


def _known_cleared(world, ctx):
    return set(ctx.cleared) | set(world.cleared_known)


def vote_obs(world, voter, ctx) -> np.ndarray:
    n, t = world.n, world.time
    pub = getattr(ctx, "pub", None)
    if pub is None:
        pub = ctx.pub = public_testimony(world)
    cleared = _known_cleared(world, ctx)
    alive = world.alive_players()
    accused = world.imp if ctx.witnesses else -1          # témoignage direct (honnête), comme la règle de base
    mine = pub[voter]
    cache = getattr(ctx, "vict", None)
    if cache is None:
        cache = ctx.vict = {}
    if voter not in cache:
        cache[voter] = victim_testimony(world, ctx, voter)
    vpub, vpriv, fresh = cache[voter]
    vmine = vpub[voter]
    g = np.array([len(alive) / n, t / world.rt.max_time, world.tasks_fraction(), min(len(ctx.bodies), 3) / 3.0,
                  1.0 if ctx.reporter == voter else 0.0, 1.0 if voter in cleared else 0.0,
                  1.0 if accused >= 0 else 0.0, 1.0 if accused == voter else 0.0,
                  sum(1 for p in alive if p not in cleared) / n,
                  _body_dist(world, world.x[voter], world.y[voter]),
                  1.0 if mine is not None else 0.0, mine[0] if mine else 1.0, mine[1] if mine else 1.0,
                  1.0 if fresh is not None else 0.0, fresh if fresh is not None else 1.0,
                  1.0 if vmine is not None else 0.0, vmine[0] if vmine else 1.0, vmine[1] if vmine else 0.0],
                 dtype=np.float32)
    oth = np.zeros((n - 1, VOTE_F), dtype=np.float32)
    for k in range(1, n):
        q = (voter + k) % n
        row = oth[k - 1]
        row[0] = 1.0 if world.alive[q] else 0.0
        row[1] = 1.0 if q in cleared else 0.0
        row[2] = 1.0 if q == accused else 0.0
        row[3] = 1.0 if q == ctx.reporter else 0.0
        pq = pub[q]
        row[4] = 1.0 if pq is not None else 0.0
        row[5], row[6] = pq if pq is not None else (1.0, 1.0)
        seen = world.mem_t[voter][q]
        if t - seen < AGE_MEM:
            row[7] = (t - seen) / AGE_MEM
            row[8] = _body_dist(world, world.mem_x[voter][q], world.mem_y[voter][q])
        else:
            row[7] = row[8] = 1.0
        row[9] = 1.0 if q in world.saw_kill[voter] else 0.0
        if vpub[q] is not None:                      # vu avec la victime, d'après les autres
            row[10], row[11], row[12] = 1.0, vpub[q][0], vpub[q][1]
        else:
            row[11] = 1.0
        if vpriv[q] is not None:                     # vu avec la victime, par moi
            row[13], row[14] = 1.0, vpriv[q]
        else:
            row[14] = 1.0
    return np.concatenate([g, oth.ravel()])


def vote_mask(world, voter) -> np.ndarray:
    m = np.zeros(N_VOTE, dtype=bool)
    for k in range(1, world.n):
        m[k - 1] = world.alive[(voter + k) % world.n]
    m[SKIP] = True
    return m


def build(world, voter, ctx):
    """-> (observation de vote, entrée du critique, masque)."""
    o = vote_obs(world, voter, ctx)
    return o, np.concatenate([o, O.global_state(world, voter)]), vote_mask(world, voter)


def dims():
    """(dimension de l'observation de vote, dimension de l'entrée du critique)."""
    from ..env import MeetingCtx
    from ..rt_world import RTWorld
    w = RTWorld(seed=0)
    w.reset(imposter=6)
    ctx = MeetingCtx(reporter=0, bodies=[], cleared=set(), witnesses=[], imposter=6)
    o, c, _ = build(w, 0, ctx)
    return len(o), len(c)


class VoteCtrl:
    """Vote avec un acteur numpy. `vote` -> (joueur visé ou None, enregistrement ou None)."""

    def __init__(self, actor, seed: int = 0, greedy: bool = False):
        self.actor = actor
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy

    def vote(self, world, voter, ctx, record=False):
        o, crit, mask = build(world, voter, ctx)
        lg = np.where(mask, self.actor.logits(o), -1e9)
        if self.greedy:
            a, rec = int(np.argmax(lg)), None
        else:
            z = lg - lg.max()
            pr = np.exp(z)
            pr /= pr.sum()
            a = int(self.rng.choice(N_VOTE, p=pr))
            rec = dict(obs=o, crit=crit, mask=mask, act=a, logp=float(np.log(max(pr[a], 1e-12)))) if record else None
        return (None if a == SKIP else (voter + a + 1) % world.n), rec


class TeacherVote:
    """Collecte d'exemples pour le pré-entraînement supervisé : vote avec la règle heuristique, et enregistre pour
    chaque crewmate (observation de vote, case de l'imposteur). L'étiquette ne sert qu'à l'entraînement."""

    def vote(self, world, voter, ctx, record=False):
        from ..env import AmongUsEnv
        choice = AmongUsEnv.ai_vote(world, voter, ctx)
        if not record or voter == world.imp:
            return choice, None
        o, crit, mask = build(world, voter, ctx)
        label = (world.imp - voter - 1) % world.n
        return choice, dict(obs=o, crit=crit, mask=mask, act=label, logp=0.0)
