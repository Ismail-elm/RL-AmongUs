"""Matchs croisés entre deux modèles (règles actuelles) : les crewmates de l'un contre les imposteurs de l'autre.

    python tools_crossplay.py ancien.npz nouveau.npz [nb_parties]
"""
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def play(a):
    cm, im, seeds = a
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.ai.vote import VoteCtrl
    from amongus.config import Config
    from amongus.rt_world import RTWorld, run_game
    ca, _ = load_actors(cm)
    cv, _ = load_vote_actors(cm)
    _, ia = load_actors(im)
    _, iv = load_vote_actors(im)
    ic = load_claim_actor(im)
    w = 0
    for s in seeds:
        r = run_game(RTWorld(Config.players(), seed=s), PolicyCtrl(ca, seed=s + 1), PolicyCtrl(ia, seed=s + 2),
                     crew_vote=VoteCtrl(cv, seed=s + 3), imp_vote=VoteCtrl(iv, seed=s + 4),
                     imp_claim=ClaimCtrl(ic, seed=s + 5))
        w += r["winner"] == "crew"
    return w, len(seeds)


def main():
    old, new = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 660
    k = max(1, (os.cpu_count() or 2) - 1)
    print("% de victoires des crewmates (règles actuelles)")
    with mp.Pool(k) as pool:
        for cn, cm in (("anciens", old), ("nouveaux", new)):
            for iname, im in (("ancien", old), ("nouvel", new)):
                res = pool.map(play, [(cm, im, list(range(20000 + i, 20000 + n, k))) for i in range(k)])
                w, t = sum(a for a, _ in res), sum(b for _, b in res)
                print(f"  {cn:8s} crewmates contre {iname:6s} imposteur : {100 * w / t:3.0f} %", flush=True)


if __name__ == "__main__":
    main()
