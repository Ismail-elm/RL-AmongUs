"""Figures de synthèse pour la présentation (results/presentation/) :
1. courbes d'apprentissage de tout l'entraînement (tous les runs bout à bout) ;
2. classement Elo des checkpoints ;
3. robustesse (test d'exploitabilité) avant / après l'entraînement contre les exploiteurs ;
4. évolution des réunions d'urgence et des fins de partie.

    python tools_figures.py
"""
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "presentation")
RUNS = ["v5a", "v5b", "v5c", "v5d", "v5e", "v5f", "v5g", "v5h"]
LABELS = {"v5a": "déplacements", "v5b": "vote appris", "v5c": "tâches", "v5d": "urgences",
          "v5e": "rééquilibrage", "v5f": "garants", "v5g": "fins de partie", "v5h": "exploiteurs"}
CREW, IMP, GREY = "#2f7fd6", "#d6452f", "#8a93a6"


def smooth(v, k=15):
    out, acc = [], []
    for x in v:
        acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def learning_curves():
    xs, win, ent_c, ent_i, bounds, off = [], [], [], [], [], 0
    for r in RUNS:
        path = os.path.join(HERE, "runs", r, "log.csv")
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        stop = os.path.join(HERE, "runs", r, "latest.json")
        last = None
        if os.path.exists(stop):
            m = re.search(r'"iter":\s*(\d+)', open(stop, encoding="utf-8").read())
            last = int(m.group(1)) if m else None
        rows = [x for x in rows if last is None or int(x["iter"]) <= last]
        for x in rows:
            xs.append(off + int(x["iter"]))
            win.append(float(x["self_crew_win"]))
            ent_c.append(float(x["crew_ent"] or 0))
            ent_i.append(float(x["imp_ent"] or 0))
        bounds.append((off, r))
        off += int(rows[-1]["iter"])
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(xs, smooth(win), color=CREW, lw=1.8)
    ax[0].axhline(0.5, color=GREY, ls="--", lw=1)
    ax[0].set_ylabel("victoires des crewmates\n(IA contre IA, lissé)")
    ax[0].set_ylim(0.1, 0.9)
    ax[1].plot(xs, smooth(ent_c), color=CREW, lw=1.6, label="crewmates")
    ax[1].plot(xs, smooth(ent_i), color=IMP, lw=1.6, label="imposteurs")
    ax[1].set_ylabel("entropie de la politique\n(déplacements)")
    ax[1].set_xlabel("itération (128 parties chacune)")
    ax[1].legend(frameon=False)
    for a in ax:
        for x0, r in bounds:
            a.axvline(x0, color="#d9dde5", lw=1)
        a.spines[["top", "right"]].set_visible(False)
    for x0, r in bounds:
        ax[0].text(x0 + 15, 0.85, LABELS[r], fontsize=8, color="#555", rotation=90, va="top")
    fig.suptitle("Apprentissage sur tout l'entraînement (~510 000 parties)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1_apprentissage.png"), dpi=150)


def elo(md):
    rows = [l for l in open(md, encoding="utf-8") if l.startswith("| v5")]
    it, ec, ei = [], [], []
    for l in rows:
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        it.append(int(c[1]))
        ec.append(int(c[2]))
        ei.append(int(c[3]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(it, ec, "o-", color=CREW, lw=2, label="crewmates")
    ax.plot(it, ei, "s-", color=IMP, lw=2, label="imposteurs")
    ax.axhline(1000, color=GREY, ls="--", lw=1)
    ax.text(it[0], 1008, "imposteur scripté (référence 1000)", fontsize=8, color="#555")
    ax.set_xlabel("itération")
    ax.set_ylabel("Elo")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Classement Elo des checkpoints (règles finales, 9 180 parties)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2_elo.png"), dpi=150)


def bars(fname, title, groups, ylabel, ref=None, fmt="{:.0f} %"):
    """groups : [(étiquette, [(nom, valeur, couleur), ...]), ...]"""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x, ticks, labels = 0, [], []
    for lab, vals in groups:
        start = x
        for name, v, col in vals:
            ax.bar(x, v, color=col, width=0.8)
            ax.text(x, v + 1, fmt.format(v), ha="center", fontsize=9)
            ax.text(x, -6, name, ha="center", fontsize=8, color="#444")
            x += 1
        ticks.append((start + x - 1) / 2)
        labels.append(lab)
        x += 0.8
    if ref is not None:
        ax.axhline(ref, color=GREY, ls="--", lw=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    learning_curves()
    md = os.path.join(HERE, "results", "v5h", "elo.md")
    if os.path.exists(md):
        elo(md)
    # chiffres mesurés (tools_measure / tools_endgame / train_mappo --train, graines fixes) : voir README
    bars("3_robustesse.png", "Robustesse : un adversaire entraîné SPÉCIALEMENT contre l'IA, combien gagne-t-il ?",
         [("imposteurs spécialisés\ncontre nos crewmates", [("avant", 54 - 34, GREY), ("après", 60 - 50, CREW)]),
          ("crewmates spécialisés\ncontre nos imposteurs", [("avant", 80 - 54, GREY), ("après", 63 - 60, IMP)])],
         "points de victoire gagnés\npar l'exploiteur", fmt="+{:.0f} pts")
    bars("4_reunions.png", "Réunions d'urgence et fins de partie : avant / après les corrections",
         [("urgences appelées\npar un imposteur", [("avant", 77, GREY), ("après", 10, CREW)]),
          ("réunions où un\ninnocent est éjecté", [("avant", 24, GREY), ("après", 14, CREW)]),
          ("fin à 2 contre 1 :\ndurée (s)", [("avant", 56, GREY), ("après", 24, IMP)])],
         "%  (ou secondes)", fmt="{:.0f}")
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
