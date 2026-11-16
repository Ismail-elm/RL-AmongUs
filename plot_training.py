#!/usr/bin/env python3
"""Courbes d'entraînement MAPPO à partir de runs/<nom>/log.csv.

    python plot_training.py runs/main    ->  results/training_curves.png
    python plot_training.py runs/vote    ->  results/training_curves_vote.png   (un fichier par run)
Si le run a un vote appris, un 4e graphique montre les éjections par partie.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs", "main")
name = os.path.basename(os.path.normpath(run))
out = os.path.join(HERE, "results", "training_curves.png" if name == "main" else f"training_curves_{name}.png")
rows = list(csv.DictReader(open(os.path.join(run, "log.csv"), encoding="utf-8")))
it = [int(r["iter"]) for r in rows]
has_vote = any((r.get("crew_vote_ent") or "0") not in ("", "0", "0.0") for r in rows)


def smooth(v, k=15):
    res = []
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]
        res.append(sum(w) / len(w))
    return res


def col(key):
    return smooth([float(r[key]) for r in rows])


n = 4 if has_vote else 3
fig, ax = plt.subplots(1, n, figsize=(5.4 * n, 4.2))
ax[0].plot(it, col("self_crew_win"), color="#3b82f6")
ax[0].set(title="Auto-jeu : victoires des crewmates (lissé)", xlabel="itération", ylim=(0, 1))
ax[0].axhline(0.5, color="gray", ls=":")
ev = [r for r in rows if r["ev_crew_vs_scripted_imp"] != ""]
x = [int(r["iter"]) for r in ev]
for key, lab in (("ev_crew_vs_scripted_imp", "crew MAPPO vs imposteur scripté"),
                 ("ev_crew_vs_prudent_imp", "crew MAPPO vs imposteur prudent"),
                 ("ev_scripted_crew_vs_imp", "crew scripté vs imposteur MAPPO"),
                 ("ev_net_vs_net", "MAPPO vs MAPPO")):
    ax[1].plot(x, [float(r[key]) for r in ev], marker="o", ms=3, label=lab)
ax[1].set(title="Évaluations (victoires des crewmates)", xlabel="itération", ylim=(0, 1))
ax[1].legend(fontsize=8)
la, = ax[2].plot(it, col("mean_sec"), color="#10b981")
ax2 = ax[2].twinx()
lb, = ax2.plot(it, col("kills"), color="#ef4444")
ax[2].set(title="Durée des parties et kills par partie", xlabel="itération")
ax[2].legend([la, lb], ["durée (s)", "kills par partie"], fontsize=8)
if has_vote:
    ax[3].plot(it, col("good_eject"), color="#10b981", label="imposteur éjecté")
    ax[3].plot(it, col("wrong_eject"), color="#ef4444", label="innocent éjecté")
    ax[3].plot(it, [v / 7 for v in col("skipped_votes")], color="#9ca3af", ls="--", label="votes blancs (÷7)")
    ax[3].set(title="Vote appris : éjections par partie (lissé)", xlabel="itération", ylim=(0, 1))
    ax[3].legend(fontsize=8)
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print("ok ->", out)
