"""Équilibre IA contre IA selon quelques réglages des règles (sans réentraîner) : aide à choisir un réglage.

    python tools_balance.py modèle.npz nb_parties "kill_cooldown_s=7" "crew_vision_range=420" ...
Chaque argument est un réglage (plusieurs séparés par des virgules) ; « base » = règles actuelles.
"""
import multiprocessing as mp
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse(spec):
    out = {}
    if spec == "base":
        return out
    for kv in spec.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def play(args):
    model, spec, seeds = args
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.ai.vote import VoteCtrl
    from amongus.config import Config
    from amongus.rt_world import RTConfig, RTWorld, run_game
    ca, ia = load_actors(model)
    cv, iv = load_vote_actors(model)
    ic = load_claim_actor(model)
    kw = parse(spec)
    cfg_kw = {k: v for k, v in kw.items() if k in ("recall",)}
    rt_kw = {k: v for k, v in kw.items() if k not in cfg_kw}
    c = Counter()
    for s in seeds:
        cfg = Config.players()
        for k, v in cfg_kw.items():
            setattr(cfg, k, v)
        w = RTWorld(cfg, RTConfig(**rt_kw), seed=s)
        r = run_game(w, PolicyCtrl(ca, seed=s + 1), PolicyCtrl(ia, seed=s + 2), crew_vote=VoteCtrl(cv, seed=s + 3),
                     imp_vote=VoteCtrl(iv, seed=s + 4), imp_claim=ClaimCtrl(ic, seed=s + 5))
        c["n"] += 1
        c[r["winner"]] += 1
        c["r_" + r["reason"]] += 1
        c["sec"] += r["seconds"]
        c["kills"] += r["kills"]
    return c


def main():
    model, n = sys.argv[1], int(sys.argv[2])
    k = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(k) as pool:
        for spec in sys.argv[3:]:
            tot = Counter()
            for c in pool.map(play, [(model, spec, list(range(i, n, k))) for i in range(k)]):
                tot.update(c)
            g = tot["n"]
            ci = 1.96 * ((tot["crew"] / g) * (1 - tot["crew"] / g) / g) ** 0.5
            print(f"{spec:40s} crew {100 * tot['crew'] / g:3.0f} % ±{100 * ci:.0f} | {tot['sec'] / g:4.0f} s | kills "
                  f"{tot['kills'] / g:.2f} | " + ", ".join(f"{k[2:]} {100 * v / g:.0f} %" for k, v in
                                                          sorted(tot.items()) if k.startswith("r_")), flush=True)


if __name__ == "__main__":
    main()
