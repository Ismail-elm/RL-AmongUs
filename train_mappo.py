#!/usr/bin/env python3
"""Entraînement MAPPO des IA (crewmates partagés + imposteur) dans le monde temps réel.

    python train_mappo.py --iters 400 --games 64 --out runs/mappo1
    python train_mappo.py --iters 600 --games 128 --out runs/vote --init runs/main/latest.pt   # ajoute le vote appris

Chaque itération : collecte de parties en parallèle (numpy), puis mise à jour PPO (PyTorch) des deux rôles.
Auto-jeu + « ligue » : une partie des parties oppose le dernier réseau à d'anciens réseaux ou à des bots scriptés.
Avec `--vote learned` (défaut), chaque rôle a aussi un réseau de VOTE (réunions), entraîné de la même façon.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import random
import time

from amongus.ai import obs as O
from amongus.ai import claim as C
from amongus.ai import vote as V
from amongus.ai.rollout import play_batch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_IMP = ["scripted", "prudent", "random"]
SCRIPT_CREW = ["scripted", "random"]


def chunk(specs, n):
    return [specs[i:i + n] for i in range(0, len(specs), n)]


def run_specs(pool, specs, greedy=False, per=4):
    out = list(pool.imap_unordered(play_batch, [dict(specs=c, greedy=greedy) for c in chunk(specs, per)]))
    return tuple([x for o in out for x in o[k]] for k in range(6))   # stats, traj crew/imp, votes crew/imp, décl.


def winrate(stats):
    return sum(s["winner"] == "crew" for s in stats) / max(1, len(stats))


def mean(stats, key):
    return sum(s[key] for s in stats) / max(1, len(stats))


def evaluate(pool, crew_p, imp_p, rng, n=40, crew_v=None, imp_v=None, imp_c=None):
    """crew_v / imp_v / imp_c : paramètres des réseaux de vote et de déclaration (None -> règles heuristiques)."""
    res = {}
    cv, iv, ic = ("net", crew_v), ("net", imp_v), ("net", imp_c)
    cases = {
        "crew_vs_scripted_imp": (("net", crew_p), cv, ("script", "scripted"), None, None),
        "crew_vs_prudent_imp": (("net", crew_p), cv, ("script", "prudent"), None, None),
        "scripted_crew_vs_imp": (("script", "scripted"), None, ("net", imp_p), iv, ic),
        "net_vs_net": (("net", crew_p), cv, ("net", imp_p), iv, ic),
    }
    for name, (c, cvote, i, ivote, iclaim) in cases.items():
        specs = [dict(seed=rng.randrange(10 ** 9), crew=c, imp=i, crew_vote=cvote, imp_vote=ivote, imp_claim=iclaim)
                 for _ in range(n)]
        res[name] = winrate(run_specs(pool, specs, per=5)[0])
    return res


def add_snap(snaps, snap, it, every, recent=8, cap=24):
    """Ligue : garde les `recent` derniers snapshots, plus un ancien sur cinq (jusqu'à `cap` au total), pour que les
    réseaux continuent d'affronter des adversaires d'il y a longtemps et n'oublient pas comment les battre."""
    snaps = snaps + [dict(snap, it=it)]
    old = [s for s in snaps[:-recent] if isinstance(s, dict) and s.get("it", 0) % (5 * every) == 0]
    return (old + snaps[-recent:])[-cap:]


def snap_parts(s):
    """Snapshot de ligue -> (déplacement, vote | None, déclaration | None)."""
    return s["act"], s.get("vote"), s.get("claim")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--games", type=int, default=64, help="parties par itération")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="processus de collecte (défaut : cœurs - 1)")
    ap.add_argument("--players", default="5+2", help="crewmates+imposteurs, ex. 5+2 ou 6+2 (10 joueurs au plus)")
    ap.add_argument("--device", default="auto", help="cpu, cuda ou auto (GPU pour la mise à jour PPO)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--ent-end", type=float, default=0.003)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--league-size", type=int, default=24, help="nb max d'anciens réseaux gardés comme adversaires")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "mappo"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--vote", choices=["learned", "scripted"], default="learned",
                    help="learned : réseaux de vote entraînés ; scripted : règle heuristique (ancien comportement)")
    ap.add_argument("--vote-warmup", type=int, default=0,
                    help="nb d'itérations initiales où seuls les réseaux de vote apprennent (déplacements figés)")
    ap.add_argument("--vote-pretrain", type=int, default=0,
                    help="nb de parties pour pré-entraîner le vote des crewmates à désigner l'imposteur (supervisé)")
    ap.add_argument("--train", choices=["both", "imp", "crew"], default="both",
                    help="camp(s) qui apprennent ; l'autre reste figé (pour mesurer la meilleure réponse d'un camp)")
    ap.add_argument("--init", default=None,
                    help="fichier latest.pt d'un autre run pour initialiser les réseaux (repart de l'itération 0)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ["AMONG_PLAYERS"] = args.players     # lu par les processus de collecte
    ctx = mp.get_context(os.environ.get("AMONG_MP") or ("fork" if "fork" in mp.get_all_start_methods() else "spawn"))
    pool = ctx.Pool(args.workers)           # créé AVANT d'importer torch (workers sans torch)
    import torch
    from amongus.ai.mappo import RoleLearner
    from amongus.ai.policy import save_actors
    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    print("mise à jour PPO sur", dev, "| collecte :", args.workers, "processus", flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    oc, oi, cc, ci = O.dims()
    crew = RoleLearner(oc, cc, args.hidden, args.lr, device=dev)
    imp = RoleLearner(oi, ci, args.hidden, args.lr, device=dev)
    learn_vote = args.vote == "learned"
    crew_v = imp_v = imp_c = None
    if learn_vote:     # vote appris (les deux camps) + déclaration apprise de l'imposteur
        vo, vc = V.dims()
        crew_v = RoleLearner(vo, vc, args.hidden, args.lr, device=dev, n_actions=V.N_VOTE)
        imp_v = RoleLearner(vo, vc, args.hidden, args.lr, device=dev, n_actions=V.N_VOTE)
        co, cc2 = C.dims()
        imp_c = RoleLearner(co, cc2, args.hidden, args.lr, device=dev, n_actions=C.N_CLAIM)
    snaps_crew, snaps_imp = [], []
    it0 = 0
    ck = os.path.join(args.out, "latest.pt")
    resuming = args.resume and os.path.exists(ck)
    if resuming or args.init:
        st = torch.load(ck if resuming else args.init, weights_only=False, map_location=dev)
        crew.load(st["crew"]); imp.load(st["imp"])
        if learn_vote and st.get("crew_vote"):
            if st["crew_vote"]["actor"]["0.weight"].shape[1] == vo:
                crew_v.load(st["crew_vote"]); imp_v.load(st["imp_vote"])
            else:
                print("réseaux de vote d'une ancienne version (autre observation) : ils repartent de zéro", flush=True)
        if learn_vote and st.get("imp_claim"):
            imp_c.load(st["imp_claim"])
        if resuming:
            snaps_crew, snaps_imp, it0 = st["snaps_crew"], st["snaps_imp"], st["it"]
            print("reprise à l'itération", it0)
        else:
            print("réseaux initialisés depuis", args.init, "(itération", st["it"], ")")
    if learn_vote and args.vote_pretrain > 0 and it0 == 0:
        # Un vote qui part de zéro apprend à toujours passer (voter au hasard coûte cher). On lui apprend donc d'abord
        # à désigner le joueur le plus suspect ; PPO apprendra ensuite QUAND voter et quand passer.
        cp, ip = crew.actor_params(), imp.actor_params()
        specs = [dict(seed=rng.randrange(10 ** 9), crew=("net", cp), imp=("net", ip), crew_vote=("teacher", None),
                      rec_crew=True) for _ in range(args.vote_pretrain)]
        cvt = run_specs(pool, specs)[3]
        acc, nex = crew_v.imitate(cvt)
        print(f"pré-entraînement du vote : {nex} exemples, imposteur désigné correctement {100 * acc:.0f} % du temps "
              f"(hasard : ~{100 / 5:.0f} %)", flush=True)
    logf = open(os.path.join(args.out, "log.csv"), "a", newline="")
    wr = csv.writer(logf)
    if it0 == 0:
        wr.writerow(["iter", "games_total", "self_crew_win", "mean_sec", "kills", "meetings", "crew_ent", "imp_ent",
                     "crew_kl", "imp_kl", "ev_crew_vs_scripted_imp", "ev_crew_vs_prudent_imp",
                     "ev_scripted_crew_vs_imp", "ev_net_vs_net", "t_collect", "t_update",
                     "good_eject", "wrong_eject", "skipped_votes", "crew_vote_ent", "imp_vote_ent", "imp_claim_ent",
                     "emergencies"])
    total = it0 * args.games
    for it in range(it0 + 1, args.iters + 1):
        frac = it / args.iters
        lr = args.lr + (args.lr_end - args.lr) * frac
        ent = args.ent + (args.ent_end - args.ent) * frac
        cp, ip = crew.actor_params(), imp.actor_params()
        cvp = crew_v.actor_params() if learn_vote else None
        ivp = imp_v.actor_params() if learn_vote else None
        icp = imp_c.actor_params() if learn_vote else None
        cv, iv, ic = ("net", cvp), ("net", ivp), ("net", icp)

        def opp(snaps, scripts):
            """-> (déplacement, vote, déclaration) d'un adversaire : ancien snapshot ou bot scripté."""
            if snaps and rng.random() < 0.5:
                act, vote, claim = snap_parts(rng.choice(snaps))
                return ("net", act), ("net", vote), ("net", claim)
            return ("script", rng.choice(scripts)), None, None

        specs = []
        for _ in range(args.games):
            u, seed = rng.random(), rng.randrange(10 ** 9)
            if u < 0.55:
                specs.append(dict(seed=seed, crew=("net", cp), imp=("net", ip), crew_vote=cv, imp_vote=iv, imp_claim=ic,
                                  rec_crew=True, rec_imp=True, tag="self"))
            elif u < 0.77:
                o, ov, oc_ = opp(snaps_imp, SCRIPT_IMP)
                specs.append(dict(seed=seed, crew=("net", cp), imp=o, crew_vote=cv, imp_vote=ov, imp_claim=oc_,
                                  rec_crew=True, tag="vs_imp_opp"))
            else:
                o, ov, _ = opp(snaps_crew, SCRIPT_CREW)
                specs.append(dict(seed=seed, crew=o, imp=("net", ip), crew_vote=ov, imp_vote=iv, imp_claim=ic,
                                  rec_imp=True, tag="vs_crew_opp"))
        t0 = time.time()
        stats, ctr, itr, cvt, ivt, ict = run_specs(pool, specs)
        t1 = time.time()
        warm = learn_vote and it <= args.vote_warmup      # échauffement : un vote aléatoire ne doit pas abîmer le reste
        tc, ti = args.train in ("both", "crew"), args.train in ("both", "imp")
        cs = {} if (warm or not tc) else crew.update(ctr, ent=ent, lr=lr, mb=2048)
        is_ = {} if (warm or not ti) else imp.update(itr, ent=ent, lr=lr, mb=256)
        cvs = crew_v.update(cvt, ent=ent, lr=lr, mb=256) if (learn_vote and tc) else {}
        ivs = imp_v.update(ivt, ent=ent, lr=lr, mb=64) if (learn_vote and ti) else {}
        ics = imp_c.update(ict, ent=ent, lr=lr, mb=64) if (learn_vote and ti) else {}
        t2 = time.time()
        total += args.games
        selfs = [s for s in stats if s["tag"] == "self"]
        row = [it, total, round(winrate(selfs), 3), round(sum(s["seconds"] for s in selfs) / max(1, len(selfs)), 1),
               round(sum(s["kills"] for s in selfs) / max(1, len(selfs)), 2),
               round(sum(s["meetings"] for s in selfs) / max(1, len(selfs)), 2),
               round(cs.get("ent", 0), 3), round(is_.get("ent", 0), 3), round(cs.get("kl", 0), 4), round(is_.get("kl", 0), 4)]
        ev = ["", "", "", ""]
        if it % args.eval_every == 0 or it == args.iters:
            r = evaluate(pool, cp, ip, rng, crew_v=cvp, imp_v=ivp, imp_c=icp)
            ev = [round(r["crew_vs_scripted_imp"], 2), round(r["crew_vs_prudent_imp"], 2),
                  round(r["scripted_crew_vs_imp"], 2), round(r["net_vs_net"], 2)]
        vrow = [round(mean(selfs, "good_ejections"), 3), round(mean(selfs, "wrong_ejections"), 3),
                round(mean(selfs, "skipped_votes"), 2), round(cvs.get("ent", 0), 3), round(ivs.get("ent", 0), 3),
                round(ics.get("ent", 0), 3), round(mean(selfs, "emergencies"), 2)]
        wr.writerow(row + ev + [round(t1 - t0, 1), round(t2 - t1, 1)] + vrow)
        logf.flush()
        print(f"it {it:4d} | self crew win {row[2]:.2f} | {row[3]:5.0f}s kills {row[4]:.2f} meet {row[5]:.2f} | "
              f"ent c/i {row[6]:.2f}/{row[7]:.2f} kl {row[8]:.3f}/{row[9]:.3f} | "
              f"éject. bonnes/mauvaises {vrow[0]:.2f}/{vrow[1]:.2f} ent vote {vrow[3]:.2f}/{vrow[4]:.2f} "
              f"décl. {vrow[5]:.2f} urgences {vrow[6]:.2f} | "
              f"collect {t1 - t0:.0f}s upd {t2 - t1:.0f}s"
              + (f" | EVAL c-vs-scr {ev[0]} c-vs-prud {ev[1]} scrCrew-vs-imp {ev[2]} net/net {ev[3]}" if ev[0] != "" else ""),
              flush=True)
        if it % args.snap_every == 0:
            snaps_crew = add_snap(snaps_crew, dict(act=crew.actor_params(), vote=cvp), it, args.snap_every,
                                  cap=args.league_size)
            snaps_imp = add_snap(snaps_imp, dict(act=imp.actor_params(), vote=ivp, claim=icp), it, args.snap_every,
                                 cap=args.league_size)
        votes = dict(crew_vote=crew_v.actor_params(), imp_vote=imp_v.actor_params(),
                     imp_claim=imp_c.actor_params()) if learn_vote else {}
        meta = dict(iter=it, games=total, hidden=args.hidden, vote=args.vote, players=args.players)
        save_actors(os.path.join(args.out, "latest.npz"), crew.actor_params(), imp.actor_params(), meta=meta, **votes)
        if it % 25 == 0:
            save_actors(os.path.join(args.out, f"ckpt_{it:04d}.npz"), crew.actor_params(), imp.actor_params(),
                        meta=meta, **votes)
        if it % 5 == 0 or it == args.iters:
            torch.save(dict(crew=crew.state(), imp=imp.state(), snaps_crew=snaps_crew, snaps_imp=snaps_imp, it=it,
                            crew_vote=crew_v.state() if learn_vote else None,
                            imp_vote=imp_v.state() if learn_vote else None,
                            imp_claim=imp_c.state() if learn_vote else None), ck)
    pool.close()


if __name__ == "__main__":
    main()
