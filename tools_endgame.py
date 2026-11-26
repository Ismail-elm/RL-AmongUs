"""Fins de partie (IA contre IA) : la partie démarre directement en fin de partie, on regarde qui gagne.

    python tools_endgame.py [modèle.npz] [nb_parties]
"""
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def play(args):
    model, seeds = args
    from amongus import rt_map as M
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.ai.vote import VoteCtrl
    from amongus.config import Config
    from amongus.rt_world import RTWorld
    ca, ia = load_actors(model)
    cv, iv = load_vote_actors(model)
    ic = load_claim_actor(model)
    out = []
    for s in seeds:
        w = RTWorld(Config.players(), seed=s)
        w.reset()
        w.set_controllers(PolicyCtrl(ca, seed=s + 1), PolicyCtrl(ia, seed=s + 2), crew_vote=VoteCtrl(cv, seed=s + 3),
                          imp_vote=VoteCtrl(iv, seed=s + 4), imp_claim=ClaimCtrl(ic, seed=s + 5))
        for p in range(w.n):
            assert M.can_stand(w.x[p], w.y[p])
        nc, ni = w.start_endgame()
        t0, k0, v0 = w.time, w.stats["kills"], w.stats.get("vents", 0)
        vents = 0
        while w.winner is None and w.time < 2000:
            w.update(0.1)
            vents += sum(1 for e in w.drain_events() if e["type"] == "vent")
        out.append((f"{nc}v{ni}", w.winner, w.reason, w.time - t0, w.stats["kills"] - k0, vents))
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "models/mappo.npz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 440
    k = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(k) as pool:
        res = [x for r in pool.map(play, [(model, list(range(10 ** 6 + i, 10 ** 6 + n, k))) for i in range(k)]) for x in r]
    print(f"== fins de partie, {model} ({len(res)} parties)")
    by = defaultdict(list)
    for r in res:
        by[r[0]].append(r)
    for key in sorted(by):
        sub = by[key]
        c = Counter((x[1], x[2]) for x in sub)
        print(f"  {key} ({len(sub):3d}) : imposteurs gagnent {100 * sum(1 for x in sub if x[1] == 'imp') / len(sub):3.0f} %"
              f" | durée {sum(x[3] for x in sub) / len(sub):4.0f} s | conduits/partie {sum(x[5] for x in sub) / len(sub):4.1f}"
              f" | " + ", ".join(f"{a}/{b} {100 * v / len(sub):.0f} %" for (a, b), v in c.most_common()))


if __name__ == "__main__":
    main()
