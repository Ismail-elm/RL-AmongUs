#!/usr/bin/env python3
"""Matrice d'évaluation : équipe (MAPPO / scriptée / aléatoire) contre imposteur (MAPPO / scripté / prudent / aléatoire).

    python evaluate_mappo.py --model models/mappo.npz --n 200
Écrit results/evaluation.md. Chaque case = % de victoires des crewmates (intervalle de confiance à 95 %).
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random

from amongus.ai.policy import load_actors, load_claim_actor, load_vote_actors
from amongus.ai.rollout import play_batch

HERE = os.path.dirname(os.path.abspath(__file__))


def params(actor):
    return actor.params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "mappo.npz"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--players", default="5+2", help="crewmates+imposteurs, ex. 5+2 ou 6+2 (10 joueurs au plus)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    crew_a, imp_a = load_actors(args.model)
    cva, iva = load_vote_actors(args.model)       # None si le modèle n'a pas de vote appris
    votes = {"Crew MAPPO": ("net", cva.params) if cva else None, "Imp MAPPO": ("net", iva.params) if iva else None}
    ica = load_claim_actor(args.model)            # None : déclaration heuristique
    crews = {"Crew MAPPO": ("net", params(crew_a)), "Crew scripté": ("script", "scripted"),
             "Crew aléatoire": ("script", "random")}
    imps = {"Imp MAPPO": ("net", params(imp_a)), "Imp scripté": ("script", "scripted"),
            "Imp prudent": ("script", "prudent"), "Imp aléatoire": ("script", "random")}
    rng = random.Random(123)
    os.environ["AMONG_PLAYERS"] = args.players
    pool = mp.get_context(os.environ.get("AMONG_MP") or ("fork" if "fork" in mp.get_all_start_methods() else "spawn")).Pool(args.workers)
    lines = ["| Équipe \\ Imposteur | " + " | ".join(imps) + " |", "|---|" + "---|" * len(imps)]
    for cn, c in crews.items():
        cells = []
        for inn, i in imps.items():
            specs = [dict(seed=rng.randrange(10 ** 9), crew=c, imp=i, crew_vote=votes.get(cn), imp_vote=votes.get(inn),
                          imp_claim=("net", ica.params) if (ica is not None and inn == "Imp MAPPO") else None)
                     for _ in range(args.n)]
            tasks = [dict(specs=specs[k:k + 5]) for k in range(0, len(specs), 5)]
            st = [s for part in pool.map(play_batch, tasks) for s in part[0]]
            p = sum(s["winner"] == "crew" for s in st) / len(st)
            ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(st))
            kills = sum(s["kills"] for s in st) / len(st)
            cells.append(f"{100 * p:.0f}% ±{100 * ci:.0f} ({kills:.1f} kills)")
            print(cn, "vs", inn, cells[-1], flush=True)
        lines.append(f"| {cn} | " + " | ".join(cells) + " |")
    os.makedirs(args.out, exist_ok=True)
    txt = f"# Évaluation (victoires des crewmates, {args.n} parties/case, IC 95 %)\n\n" + "\n".join(lines) + "\n"
    open(os.path.join(args.out, "evaluation.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
