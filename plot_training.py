#!/usr/bin/env python3
"""Courbes d'entraînement MAPPO à partir de runs/<nom>/log.csv  ->  results/training_curves.png"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs", "main")
rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
it = [int(r["iter"]) for r in rows]


def smooth(v, k=15):
    out = []
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(it, smooth([float(r["self_crew_win"]) for r in rows]), color="#3b82f6")
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
ax[2].plot(it, smooth([float(r["mean_sec"]) for r in rows]), label="durée (s)", color="#10b981")
ax2 = ax[2].twinx()
ax2.plot(it, smooth([float(r["kills"]) for r in rows]), label="kills", color="#ef4444")
ax[2].set(title="Durée des parties et kills par partie", xlabel="itération")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
fig.savefig(os.path.join(HERE, "results", "training_curves.png"), dpi=130)
print("ok")
