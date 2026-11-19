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
from ..env import DIST, FIX, KILL, N_ACTIONS, N_ROOMS, NONE_HINT, TASK, WAIT
from ..mapdata import NEXT_HOP

MAX_PLAYERS = 10                # les réseaux sont dimensionnés pour 10 joueurs ; cases vides s'il y en a moins
N_PLAYERS = MAX_PLAYERS         # (nom historique)
W, H = float(M.WORLD_W), float(M.WORLD_H)
OTHER_F = 11                    # nb de features par autre joueur
N_OTHERS = MAX_PLAYERS - 1


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
    R = world.vision(p)
    is_imp = p in world.imps
    x, y = world.x[p], world.y[p]
    room = world.room[p]
    crew = world.crew_alive()
    # nombre de joueurs que p croit vivants (un crewmate n'apprend une mort qu'en réunion ou en la voyant)
    n_alive = len(crew) if is_imp else sum(1 for q in range(n) if q != p and world.believes_alive(p, q))
    f = [np.array([x / W, y / H, t / rt.max_time, world.tasks_fraction(), n_alive / N_OTHERS,
                   1.0 if room < 0 else 0.0, float(world.emergencies_left[p] > 0),
                   1.0 if world.can_emergency(p) else 0.0, min(60.0, t - world.last_meeting_t) / 60.0,
                   # lumières : coupées ? depuis combien de temps ? imposteur : puis-je saboter ? crewmate : réparer ici ?
                   1.0 if world.lights_off else 0.0,
                   min(1.0, (t - world.lights_t) / rt.lights_max) if world.lights_off else 0.0,
                   1.0 if (is_imp and world.can_sabotage()) else 0.0,
                   1.0 if (not is_imp and world.lights_off and room == M.FIX_ROOM) else 0.0],
                  dtype=np.float32), _onehot(room, N_ROOMS)]
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
        fix_hint = NEXT_HOP[room][M.FIX_ROOM] if (world.lights_off and room >= 0) else NONE_HINT   # vers Électricité
        f += [tv, np.array([len(ts) / 5.0, 1.0 if room in ts else 0.0,
                            min(60.0, t - world.ag[p]["prog_t"]) / 60.0, dmin], dtype=np.float32),
              _onehot(hint, N_ROOMS + 1), _onehot(fix_hint, N_ROOMS + 1)]
    else:
        seeing_me = sum(1 for q in crew if world.sees(q, p))
        vis_crew = [q for q in crew if world.sees(p, q)]
        dn = min((math.hypot(world.x[q] - x, world.y[q] - y) for q in vis_crew), default=R)
        n_imp = len(world.imps)
        f.append(np.array([world.kcd[p] / rt.kill_cooldown_s, 1.0 if world.kcd[p] <= 0 else 0.0,
                           seeing_me / N_OTHERS, len(vis_crew) / N_OTHERS, min(dn, R) / R,
                           1.0 if dn <= rt.kill_range else 0.0,
                           len(world.imps_alive()) / 3.0, (n_imp - 1) / 2.0], dtype=np.float32))
    # les autres joueurs, en ordre relatif (p+1, p+2, ...)
    oth = np.zeros((N_OTHERS, OTHER_F), dtype=np.float32)
    for k in range(1, n):
        q = (p + k) % n
        row = oth[k - 1]
        alive = world.alive[q]
        row[6] = 1.0 if world.believes_alive(p, q) else 0.0
        row[7] = 1.0 if q in world.cleared_known else 0.0
        row[8] = 1.0 if q in world.saw_kill[p] else 0.0
        row[9] = 1.0                                   # case occupée (moins de 10 joueurs : cases vides)
        row[10] = 1.0 if (is_imp and q in world.imps) else 0.0   # partenaire (seuls les imposteurs le savent)
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
    rt, n, N = world.rt, world.n, MAX_PLAYERS
    pos = np.zeros(N * 2, dtype=np.float32)
    alive = np.zeros(N, dtype=np.float32)
    imps = np.zeros(N, dtype=np.float32)
    tasks = np.zeros((N, N_ROOMS), dtype=np.float32)
    cleared = np.zeros(N, dtype=np.float32)
    for q in range(n):
        pos[2 * q] = world.x[q] / W
        pos[2 * q + 1] = world.y[q] / H
        alive[q] = 1.0 if world.alive[q] else 0.0
        for r in world.tasks[q]:
            tasks[q, r] = 1.0
        cleared[q] = 1.0 if q in world.cleared_known else 0.0
        imps[q] = 1.0 if q in world.imps else 0.0
    kc = min((world.kcd[i] for i in world.imps_alive()), default=0.0)
    misc = np.array([world.time / rt.max_time, kc / rt.kill_cooldown_s,
                     world.tasks_fraction(), min(len(world.bodies), 3) / 3.0, n / N], dtype=np.float32)
    return np.concatenate([pos, alive, imps, tasks.ravel(), cleared, misc, _onehot(p, N)]).astype(np.float32)


def build(world, p, legal):
    """-> (obs locale, entrée du critique, masque d'actions)."""
    o = local_obs(world, p)
    return o, np.concatenate([o, global_state(world, p)]), action_mask(legal)


def dims():
    """Dimensions (obs crew, obs imposteur, critique crew, critique imposteur) par construction d'un monde."""
    from ..rt_world import RTWorld
    w = RTWorld(seed=0)
    w.reset(imposter=w.n - 1)
    crew = next(q for q in range(w.n) if q not in w.imps)
    out = []
    for p in (crew, w.n - 1):
        o, c, _ = build(w, p, w.legal_actions(p))
        out.append((len(o), len(c)))
    return out[0][0], out[1][0], out[0][1], out[1][1]
