#!/usr/bin/env python3
"""Analyse d'un entraînement : classement Elo entre checkpoints, et rapport de comportement d'un modèle.

    python analyze.py elo runs/v4a runs/v4b --every 100 --n 20     # classement des checkpoints (crew et imposteur)
    python analyze.py behavior runs/v4b/latest.npz --n 400          # comment les IA jouent (conduits, sabotage, mensonges...)

Classement Elo : chaque équipe de crewmates (réseaux d'un checkpoint) affronte chaque imposteur (d'un checkpoint), plus
des bots scriptés de référence. On ajuste des notes Elo telles que P(les crewmates gagnent) = 1 / (1 + 10^((Ri - Rc)/400)).
L'imposteur scripté est fixé à 1000. Une note n'a de sens que comparée aux autres notes du même camp : elle dit si le
camp progresse vraiment, ce que le taux de victoire en auto-jeu ne dit pas (les deux camps progressent ensemble).
Écrit results/<nom>/elo.md (+ elo.png) et results/<nom>/comportement.md.
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import re
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# =================================================================== chargement
def load_side(path):
    """-> dict(crew=('net', params), crew_vote=..., imp=..., imp_vote=..., imp_claim=...) d'un fichier .npz."""
    from amongus.ai.policy import load_actors, load_claim_actor, load_vote_actors
    ca, ia = load_actors(path)
    cv, iv = load_vote_actors(path)
    ic = load_claim_actor(path)
    net = lambda a: ("net", a.params) if a is not None else None
    return dict(crew=net(ca), crew_vote=net(cv), imp=net(ia), imp_vote=net(iv), imp_claim=net(ic))


def checkpoints(runs, every):
    """Checkpoints des runs donnés, un tous les `every` itérations (+ le dernier de chaque run)."""
    out = []
    offset = 0
    for run in runs:
        files = sorted(glob.glob(os.path.join(run, "ckpt_*.npz")))
        its = [int(re.search(r"ckpt_(\d+)", f).group(1)) for f in files]
        keep = [(i, f) for i, f in zip(its, files) if i % every == 0]
        if files and (not keep or keep[-1][1] != files[-1]):
            keep.append((its[-1], files[-1]))
        name = os.path.basename(os.path.normpath(run))
        out += [(f"{name}:{i}", offset + i, f) for i, f in keep]
        offset += max(its) if its else 0
    return out


# =================================================================== Elo
def _play(task):
    from amongus.ai.rollout import play_batch
    key, specs = task
    st = play_batch(dict(specs=specs))[0]
    return key, [s["winner"] == "crew" for s in st]


def fit_elo(results, anchor, iters=4000, lr=4.0):
    """results : {(crew, imp): [victoires crew (bool)]}. -> {agent: note}. Maximum de vraisemblance (+ léger a priori)."""
    agents = sorted({a for k in results for a in k})
    idx = {a: i for i, a in enumerate(agents)}
    R = np.full(len(agents), 1000.0)
    c = np.array([idx[k[0]] for k in results])
    i = np.array([idx[k[1]] for k in results])
    w = np.array([sum(v) for v in results.values()], dtype=float)
    n = np.array([len(v) for v in results.values()], dtype=float)
    k = math.log(10) / 400
    for _ in range(iters):
        p = 1 / (1 + np.exp(-k * (R[c] - R[i])))
        g = np.zeros_like(R)
        np.add.at(g, c, (w - n * p))
        np.add.at(g, i, -(w - n * p))
        g -= (R - 1000.0) / 400 ** 2 * 4          # a priori faible : évite les notes infinies (100 % ou 0 %)
        R += lr * g / max(1.0, n.mean())
        R += 1000.0 - R[idx[anchor]]
    return {a: float(R[idx[a]]) for a in agents}


def elo(args):
    cks = checkpoints(args.runs, args.every)
    sides = {name: load_side(f) for name, _it, f in cks}
    crews = {f"crew {n}": dict(crew=s["crew"], crew_vote=s["crew_vote"]) for n, s in sides.items()}
    imps = {f"imp {n}": dict(imp=s["imp"], imp_vote=s["imp_vote"], imp_claim=s["imp_claim"]) for n, s in sides.items()}
    crews["crew scripté"] = dict(crew=("script", "scripted"), crew_vote=None)
    imps["imp scripté"] = dict(imp=("script", "scripted"), imp_vote=None, imp_claim=None)
    imps["imp prudent"] = dict(imp=("script", "prudent"), imp_vote=None, imp_claim=None)
    rng = random.Random(7)
    tasks = []
    for cn, c in crews.items():
        for inn, i in imps.items():
            specs = [dict(seed=rng.randrange(10 ** 9), **c, **i) for _ in range(args.n)]
            for k in range(0, len(specs), 5):
                tasks.append(((cn, inn), specs[k:k + 5]))
    print(f"{len(crews)} équipes × {len(imps)} imposteurs × {args.n} parties = {len(crews) * len(imps) * args.n} parties",
          flush=True)
    results = {}
    with mp.Pool(args.workers) as pool:
        for key, wins in pool.imap_unordered(_play, tasks):
            results.setdefault(key, []).extend(wins)
    R = fit_elo(results, "imp scripté")
    its = {n: it for n, it, _ in cks}
    lines = [f"# Classement Elo ({args.n} parties par paire, imposteur scripté = 1000)", "",
             "| Checkpoint | Itération | Elo crewmates | Elo imposteur | Crew vs imp. scripté | Crew scripté vs imp. |",
             "|---|---|---|---|---|---|"]
    rows = []
    for n, _it, _f in cks:
        rc, ri = R[f"crew {n}"], R[f"imp {n}"]
        wc = np.mean(results[(f"crew {n}", "imp scripté")])
        wi = np.mean(results[("crew scripté", f"imp {n}")])
        rows.append((its[n], rc, ri))
        lines.append(f"| {n} | {its[n]} | {rc:.0f} | {ri:.0f} | {100 * wc:.0f} % | {100 * wi:.0f} % |")
    lines += ["", f"Références : crew scripté {R['crew scripté']:.0f}, imposteur prudent {R['imp prudent']:.0f}.", "",
              "Lecture : +100 points d'écart entre une équipe et un imposteur = 64 % de victoires pour l'équipe ; "
              "+200 = 76 %. Comparer les notes d'un même camp entre checkpoints dit s'il progresse."]
    txt = "\n".join(lines) + "\n"
    out = os.path.join(HERE, "results", os.path.basename(os.path.normpath(args.runs[-1])))
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "elo.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x = [r[0] for r in rows]
        ax.plot(x, [r[1] for r in rows], marker="o", label="crewmates", color="#3b82f6")
        ax.plot(x, [r[2] for r in rows], marker="o", label="imposteur", color="#ef4444")
        ax.axhline(R["crew scripté"], color="#3b82f6", ls=":", lw=1, label="crew scripté")
        ax.axhline(1000, color="#ef4444", ls=":", lw=1, label="imposteur scripté")
        ax.set(title="Classement Elo des checkpoints", xlabel="itération (cumulée sur les runs)", ylabel="Elo")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "elo.png"), dpi=130)
        print("courbe :", os.path.join(out, "elo.png"))
    except ImportError:
        pass


# =================================================================== comportement
def _behave(task):
    path, seeds = task
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.ai.vote import VoteCtrl
    from amongus.rt_world import RTWorld
    ca, ia = load_actors(path)
    cv, iv = load_vote_actors(path)
    ic = load_claim_actor(path)
    g, kills, meets, lights = [], [], [], []
    for s in seeds:
        from amongus.config import Config
        w = RTWorld(Config.players(), seed=s)
        w.reset()
        w.set_controllers(PolicyCtrl(ca, seed=s + 1), PolicyCtrl(ia, seed=s + 2),
                          crew_vote=VoteCtrl(cv, seed=s + 3) if cv else None,
                          imp_vote=VoteCtrl(iv, seed=s + 4) if iv else None,
                          imp_claim=ClaimCtrl(ic, seed=s + 5) if ic else None)
        game = Counter()
        n = 0
        while w.winner is None and n < 20000:
            w.update(0.1)
            n += 1
            for e in w.drain_events():
                t = e["type"]
                if t == "kill":
                    kills.append(dict(witnessed=bool(e["witnesses"]), dark=w.lights_off))
                elif t == "vent":
                    game["vents"] += 1
                    game["vents_seen"] += bool(e["seen"])
                elif t == "vent_seen":
                    game["vents_seen_exit"] += 1
                elif t == "lights":
                    if not e["on"]:
                        game["sabotages"] += 1
                    else:
                        lights.append("réparées" if e.get("fixer") is not None else "seules")
                elif t == "meeting":
                    ctx, ej = e["ctx"], e["ejected"]
                    imps = [i for i in w.imps if i in ctx.claims or i == ej or w.alive[i]]
                    for k, imp in enumerate(imps):      # une ligne par imposteur présent à la réunion
                        claim = ctx.claims.get(imp)
                        true = ctx.ref.zone.get(imp) if ctx.ref is not None else None
                        acc = ctx.accuse.get(imp)
                        meets.append(dict(
                            emergency=ctx.emergency, by_imp=e["reporter"] == imp, ref=ctx.ref is not None and claim is not None,
                            lie=claim is not None and claim != true,
                            claim_kill_room=claim is not None and claim == ctx.kill_room,
                            caught=bool(ctx.contra.get(imp) or ctx.absent.get(imp)),
                            accused_by_wit=any(a not in w.imps and tt == imp for a, tt in ctx.accuse.items()),
                            imp_accuses=acc is not None, imp_ej=ej == imp,
                            inno_ej=ej is not None and ej not in w.imps, framed=acc is not None and ej == acc,
                            first=k == 0, any_imp_ej=ej is not None and ej in w.imps))
        g.append(dict(win=w.winner == "crew", reason=w.reason, sec=w.time, **game))
    return g, kills, meets, lights


def behavior(args):
    seeds = list(range(70000, 70000 + args.n))
    with mp.Pool(args.workers) as pool:
        parts = pool.map(_behave, [(args.model, seeds[k:k + 10]) for k in range(0, args.n, 10)])
    games = [x for p in parts for x in p[0]]
    kills = [x for p in parts for x in p[1]]
    meets = [x for p in parts for x in p[2]]
    lights = [x for p in parts for x in p[3]]
    pct = lambda xs: 100 * sum(xs) / max(1, len(xs))
    per = lambda k: sum(x.get(k, 0) for x in games) / max(1, len(games))
    G = len(games)
    lie_m = [m for m in meets if m["ref"]]
    mt = [m for m in meets if m["first"]]                  # une ligne par réunion (statistiques de réunion)
    n_sab = sum(x.get("sabotages", 0) for x in games)
    ends = Counter({"réparées par un crewmate": lights.count("réparées"), "seules (45 s)": lights.count("seules"),
                    "par une réunion ou la fin de partie": n_sab - len(lights)})
    L = [f"# Comportement : {args.model} ({G} parties IA contre IA)", "",
         "## Parties",
         f"- Crewmates gagnants : **{pct([x['win'] for x in games]):.0f} %** ; durée moyenne {per('sec'):.0f} s",
         "- Fins : " + ", ".join(f"{k} {100 * v / G:.0f} %" for k, v in Counter(x['reason'] for x in games).most_common()),
         "", "## Imposteur",
         f"- Meurtres par partie : {len(kills) / max(1, G):.2f} ; avec témoin {pct([k['witnessed'] for k in kills]):.0f} % ; "
         f"dans le noir {pct([k['dark'] for k in kills]):.0f} %",
         f"- Conduits par partie : {per('vents'):.2f} ; vu en entrant {100 * per('vents_seen') / max(per('vents'), 1e-9):.0f} %, "
         f"vu en sortant {100 * per('vents_seen_exit') / max(per('vents'), 1e-9):.0f} %",
         f"- Sabotages des lumières par partie : {per('sabotages'):.2f} ; lumières rallumées : "
         + ", ".join(f"{k} {100 * v / max(1, n_sab):.0f} %" for k, v in ends.most_common()),
         f"- Réunions déclenchées par un imposteur (corps) : {pct([m['by_imp'] and not m['emergency'] for m in meets]):.0f} % (par imposteur)",
         "", "## Réunions",
         f"- {len(mt) / max(1, G):.2f} réunions par partie, dont urgences {pct([m['emergency'] for m in mt]):.0f} %",
         f"- Un imposteur éjecté {pct([m['any_imp_ej'] for m in mt]):.0f} % "
         f"(par imposteur présent : {pct([m['imp_ej'] for m in meets]):.0f} %), innocent éjecté {pct([m['inno_ej'] for m in mt]):.0f} %",
         f"- Déclarations de l'imposteur (réunions après un meurtre, {len(lie_m)}) : ment {pct([m['lie'] for m in lie_m]):.0f} %, "
         f"se dit dans la salle du crime {pct([m['claim_kill_room'] for m in lie_m]):.0f} %, "
         f"démasqué par les recoupements {pct([m['caught'] for m in lie_m]):.0f} %, "
         f"accuse quelqu'un {pct([m['imp_accuses'] for m in lie_m]):.0f} %",
         f"- Quand il accuse quelqu'un, l'accusé est éjecté {pct([m['framed'] for m in lie_m if m['imp_accuses']]):.0f} % du temps",
         f"- Éjecté quand un témoin l'accuse : {pct([m['imp_ej'] for m in lie_m if m['accused_by_wit']]):.0f} % ; "
         f"quand ses recoupements le trahissent : {pct([m['imp_ej'] for m in lie_m if m['caught']]):.0f} % ; "
         f"sinon : {pct([m['imp_ej'] for m in lie_m if not m['caught'] and not m['accused_by_wit']]):.0f} %",
         f"- Réunions d'urgence : imposteur éjecté {pct([m['imp_ej'] for m in meets if m['emergency']]):.0f} %"]
    txt = "\n".join(L) + "\n"
    out = os.path.join(HERE, "results", os.path.basename(os.path.dirname(os.path.abspath(args.model))))
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "comportement.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("elo", help="classement Elo des checkpoints")
    e.add_argument("runs", nargs="+", help="dossiers de runs, dans l'ordre (ex. runs/v4a runs/v4b)")
    e.add_argument("--every", type=int, default=100, help="un checkpoint tous les N itérations")
    e.add_argument("--n", type=int, default=20, help="parties par paire (équipe, imposteur)")
    b = sub.add_parser("behavior", help="rapport de comportement d'un modèle")
    b.add_argument("model")
    b.add_argument("--n", type=int, default=400)
    for p in (e, b):
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--players", default="5+2", help="crewmates+imposteurs, ex. 5+2 ou 6+2")
    args = ap.parse_args()
    os.environ["AMONG_PLAYERS"] = args.players
    {"elo": elo, "behavior": behavior}[args.cmd](args)


if __name__ == "__main__":
    main()
