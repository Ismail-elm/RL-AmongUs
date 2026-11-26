"""Quels indices le vote des crewmates utilise-t-il ? On MÉLANGE un groupe d'indices entre les joueurs vivants
(crewmates seulement) : les valeurs restent plausibles mais ne désignent plus le bon joueur. On regarde la chute
du taux de victoire et des bonnes éjections.

    python tools_ablation.py modèle.npz nb_parties
"""
import multiprocessing as mp
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "aucun": [],
    "accusations": [2, 23] + list(range(24, 33)),
    "déclarations+recoupements": list(range(15, 23)),
    "mémoire (vu quand, où)": [4, 5, 6, 7, 8],
    "vu avec la victime": [10, 11, 12, 13, 14],
    "a signalé": [3],
    "témoin direct": [9],
    "alibi confirmé": [1],
}


def play(args):
    model, group, seeds = args
    import numpy as np
    from amongus.ai import vote as V
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.config import Config
    from amongus.rt_world import RTWorld, run_game
    cols = GROUPS[group]
    import random
    rng = random.Random(0)
    orig = V.vote_obs

    def masked(world, voter, ctx):
        o = orig(world, voter, ctx)
        if cols and voter not in world.imps:
            o = o.copy()
            rows = o[V.N_GLOBAL:].reshape(V.N_SLOTS, V.VOTE_F)
            occ = [i for i in range(V.N_SLOTS) if rows[i, 24 + V.N_SLOTS] > 0 and rows[i, 0] > 0]
            perm = list(occ)
            rng.shuffle(perm)                    # mêmes valeurs, attribuées au hasard aux joueurs vivants
            rows[occ, :] = rows[occ, :]
            sub = rows[np.ix_(perm, cols)].copy()
            rows[np.ix_(occ, cols)] = sub
        return o

    V.vote_obs = masked
    ca, ia = load_actors(model)
    cv, iv = load_vote_actors(model)
    ic = load_claim_actor(model)
    c = Counter()
    for s in seeds:
        w = RTWorld(Config.players(), seed=s)
        r = run_game(w, PolicyCtrl(ca, seed=s + 1), PolicyCtrl(ia, seed=s + 2), crew_vote=V.VoteCtrl(cv, seed=s + 3),
                     imp_vote=V.VoteCtrl(iv, seed=s + 4), imp_claim=ClaimCtrl(ic, seed=s + 5))
        c["n"] += 1
        c[r["winner"]] += 1
        c["good"] += r["good_ejections"]
        c["bad"] += r["wrong_ejections"]
        c["meet"] += r["meetings"]
    return c


def main():
    model, n = sys.argv[1], int(sys.argv[2])
    k = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(k) as pool:
        for g in GROUPS:
            tot = Counter()
            for c in pool.map(play, [(model, g, list(range(i, n, k))) for i in range(k)]):
                tot.update(c)
            m = max(1, tot["meet"])
            print(f"sans {g:28s} crew {100 * tot['crew'] / tot['n']:3.0f} % | imposteur éjecté "
                  f"{100 * tot['good'] / m:3.0f} % des réunions | innocent {100 * tot['bad'] / m:3.0f} %", flush=True)


if __name__ == "__main__":
    main()
