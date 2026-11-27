"""Fiabilité des chiffres : on rejoue la même mesure sur plusieurs séries de parties INDÉPENDANTES (graines
différentes) et on regarde si les résultats concordent avec l'intervalle de confiance annoncé.

    python tools_reliability.py modèle.npz [nb_séries] [parties_par_série]

Pour chaque série : victoires des crewmates IA contre IA, contre l'imposteur scripté, et de l'équipe scriptée contre
les imposteurs IA. Si l'IC à 95 % est juste, environ 95 % des séries tombent dans l'IC calculé sur l'ensemble.
"""
import math
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CASES = ("IA contre IA", "crew IA vs imposteur scripté", "crew scripté vs imposteurs IA")


def play(a):
    model, case, seeds = a
    from amongus.ai.bots import ScriptedCrew, ScriptedImposter
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.ai.vote import VoteCtrl
    from amongus.config import Config
    from amongus.rt_world import RTWorld, run_game
    ca, ia = load_actors(model)
    cv, iv = load_vote_actors(model)
    ic = load_claim_actor(model)
    w = 0
    for s in seeds:
        net_crew = case in (0, 1)
        net_imp = case in (0, 2)
        r = run_game(RTWorld(Config.players(), seed=s),
                     PolicyCtrl(ca, seed=s + 1) if net_crew else ScriptedCrew(),
                     PolicyCtrl(ia, seed=s + 2) if net_imp else ScriptedImposter(s),
                     crew_vote=VoteCtrl(cv, seed=s + 3) if net_crew else None,
                     imp_vote=VoteCtrl(iv, seed=s + 4) if net_imp else None,
                     imp_claim=ClaimCtrl(ic, seed=s + 5) if net_imp else None)
        w += r["winner"] == "crew"
    return w


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "models/mappo.npz"
    series = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    k = max(1, (os.cpu_count() or 2) - 1)
    print(f"== {model} : {series} séries indépendantes de {n} parties par cas")
    with mp.Pool(k) as pool:
        for case, name in enumerate(CASES):
            rates = []
            for b in range(series):
                base = 7_000_000 + 1_000_000 * case + 10_000 * b
                ws = pool.map(play, [(model, case, list(range(base + i, base + n, k))) for i in range(k)])
                rates.append(sum(ws) / n)
            p = sum(rates) / series
            ci_total = 1.96 * math.sqrt(p * (1 - p) / (n * series))
            ci_one = 1.96 * math.sqrt(p * (1 - p) / n)
            sd = math.sqrt(sum((r - p) ** 2 for r in rates) / max(1, series - 1))
            inside = sum(1 for r in rates if abs(r - p) <= ci_one)
            print(f"  {name:32s} : " + " ".join(f"{100 * r:3.0f}" for r in rates)
                  + f"  -> moyenne {100 * p:.1f} % ±{100 * ci_total:.1f} (IC 95 % sur {n * series} parties) ; "
                    f"écart-type entre séries {100 * sd:.1f} pts, attendu {100 * ci_one / 1.96:.1f} ; "
                    f"{inside}/{series} séries dans l'IC d'une série (±{100 * ci_one:.1f})", flush=True)


if __name__ == "__main__":
    main()
