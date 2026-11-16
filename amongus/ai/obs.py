"""Observations pour MAPPO.

* observation LOCALE (acteur) : ce que le joueur peut savoir — sa position, ses tâches, ce qu'il voit
  maintenant et ce dont il se souvient (dernière position vue de chacun, innocentés, tueurs vus).
* état GLOBAL (critique centralisé, utilisé seulement à l'entraînement) : positions, rôles et tâches de
  tout le monde. Entrée du critique = [observation locale, état global, identité du joueur].
"""
from __future__ import annotations

import math

import numpy as np

from .. import rt_map as M
from ..env import DIST, KILL, N_ACTIONS, N_ROOMS, NONE_HINT, TASK, WAIT

N_PLAYERS = 7
W, H = float(M.WORLD_W), float(M.WORLD_H)
OTHER_F = 9                     # nb de features par autre joueur
N_OTHERS = N_PLAYERS - 1


def _onehot(i, n):
    v = np.zeros(n, dtype=np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v


def action_mask(legal) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[list(legal)] = True
    return m


def local_obs(world, p) -> np.ndarray:
    rt, n, t = world.rt, world.n, world.time
    R = rt.vision_range
    is_imp = p == world.imp
    x, y = world.x[p], world.y[p]
    room = world.room[p]
    crew = world.crew_alive()
    f = [np.array([x / W, y / H, t / rt.max_time, world.tasks_fraction(), len(crew) / 6.0,
                   1.0 if room < 0 else 0.0], dtype=np.float32), _onehot(room, N_ROOMS)]
    if not is_imp:
        ts = world.tasks[p]
        tv = np.zeros(N_ROOMS, dtype=np.float32)
        for r in ts:
            tv[r] = 1.0
        dmin = 1.0
        for r in ts:
            tx, ty = M.task_terminal(r)
            dmin = min(dmin, math.hypot(tx - x, ty - y) / 2500.0)
        hint = world.crew_hint(p) if room >= 0 else NONE_HINT
        f += [tv, np.array([len(ts) / 5.0, 1.0 if room in ts else 0.0,
                            min(60.0, t - world.ag[p]["prog_t"]) / 60.0, dmin], dtype=np.float32),
              _onehot(hint, N_ROOMS + 1)]
    else:
        seeing_me = sum(1 for q in crew if world.sees(q, p))
        vis_crew = [q for q in crew if world.sees(p, q)]
        dn = min((math.hypot(world.x[q] - x, world.y[q] - y) for q in vis_crew), default=R)
        f.append(np.array([world.cd_s / rt.kill_cooldown_s, 1.0 if world.cooldown == 0 else 0.0,
                           seeing_me / 6.0, len(vis_crew) / 6.0, min(dn, R) / R,
                           1.0 if dn <= rt.kill_range else 0.0], dtype=np.float32))
    # les autres joueurs, en ordre relatif (p+1, p+2, ...)
    oth = np.zeros((N_OTHERS, OTHER_F), dtype=np.float32)
    for k in range(1, n):
        q = (p + k) % n
        row = oth[k - 1]
        alive = world.alive[q]
        row[6] = 1.0 if alive else 0.0
        row[7] = 1.0 if q in world.cleared_known else 0.0
        row[8] = 1.0 if q in world.saw_kill[p] else 0.0
        if alive and world.sees(p, q):
            row[0] = 1.0
            row[1] = (world.x[q] - x) / R
            row[2] = (world.y[q] - y) / R
        seen = world.mem_t[p][q]
        if seen > -1e8:
            row[3] = world.mem_x[p][q] / W
            row[4] = world.mem_y[p][q] / H
            row[5] = min(120.0, t - seen) / 120.0
        else:
            row[5] = 1.0
    f.append(oth.ravel())
    # cadavres visibles
    bv = np.zeros(4, dtype=np.float32)
    best = None
    for (_r, v) in world.bodies:
        bp = world.body_pos.get(v)
        if bp is None:
            continue
        d = math.hypot(bp[0] - x, bp[1] - y)
        if d < R and M.line_of_sight(x, y, bp[0], bp[1]) and (best is None or d < best[0]):
            best = (d, bp)
    if best is not None:
        bv[0] = 1.0
        bv[1] = (best[1][0] - x) / R
        bv[2] = (best[1][1] - y) / R
    bv[3] = min(len(world.bodies), 3) / 3.0
    f.append(bv)
    return np.concatenate(f).astype(np.float32)


def global_state(world, p) -> np.ndarray:
    """Information complète (critique seulement) + identité du joueur évalué."""
    rt, n = world.rt, world.n
    pos = np.zeros(n * 2, dtype=np.float32)
    alive = np.zeros(n, dtype=np.float32)
    tasks = np.zeros((n, N_ROOMS), dtype=np.float32)
    cleared = np.zeros(n, dtype=np.float32)
    for q in range(n):
        pos[2 * q] = world.x[q] / W
        pos[2 * q + 1] = world.y[q] / H
        alive[q] = 1.0 if world.alive[q] else 0.0
        for r in world.tasks[q]:
            tasks[q, r] = 1.0
        cleared[q] = 1.0 if q in world.cleared_known else 0.0
    misc = np.array([world.time / rt.max_time, world.cd_s / rt.kill_cooldown_s,
                     world.tasks_fraction(), min(len(world.bodies), 3) / 3.0], dtype=np.float32)
    return np.concatenate([pos, alive, _onehot(world.imp, n), tasks.ravel(), cleared, misc,
                           _onehot(p, n)]).astype(np.float32)


def build(world, p, legal):
    """-> (obs locale, entrée du critique, masque d'actions)."""
    o = local_obs(world, p)
    return o, np.concatenate([o, global_state(world, p)]), action_mask(legal)


def dims():
    """Dimensions (obs crew, obs imposteur, critique crew, critique imposteur) par construction d'un monde."""
    from ..rt_world import RTWorld
    w = RTWorld(seed=0)
    w.reset(imposter=n_imp if (n_imp := 6) else 6)
    out = []
    for p in (0, 6):
        o, c, _ = build(w, p, w.legal_actions(p))
        out.append((len(o), len(c)))
    return out[0][0], out[1][0], out[0][1], out[1][1]
