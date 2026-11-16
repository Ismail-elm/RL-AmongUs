#!/usr/bin/env python3
"""Joue à Among Us contre les IA entraînées (ou regarde-les jouer).

    python play.py                       # TEMPS RÉEL, tu es crewmate (ZQSD/flèches, E, K, R)
    python play.py --role imposter       # tu es l'imposteur
    python play.py --role random         # rôle tiré au sort
    python play.py --spectate            # 7 IA, tu regardes
    python play.py --crew-ai scripted    # change l'IA : mappo (apprise), scripted, random
"""
from __future__ import annotations

import argparse
import os

from amongus.ai.bots import RandomCtrl, ScriptedCrew, ScriptedImposter
from amongus.ai.policy import PolicyCtrl, load_actors, load_vote_actors
from amongus.ai.vote import VoteCtrl
from amongus.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "mappo.npz"), help="acteurs MAPPO (.npz)")
    ap.add_argument("--role", choices=["crew", "imposter", "random"], default="crew")
    ap.add_argument("--spectate", action="store_true")
    ap.add_argument("--crew-ai", choices=["mappo", "scripted", "random"], default="mappo")
    ap.add_argument("--imp-ai", choices=["mappo", "scripted", "prudent", "random"], default="mappo")
    ap.add_argument("--greedy", action="store_true", help="IA déterministes (sinon : elles tirent leur action)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--god", action="store_true", help="vue complète (debug) : voir tout le monde")
    ap.add_argument("--speed", type=float, default=1.0, help="vitesse en mode spectateur")
    args = ap.parse_args()

    crew_a = imp_a = crew_vote = imp_vote = None
    if "mappo" in (args.crew_ai, args.imp_ai):
        crew_a, imp_a = load_actors(args.model)
        cva, iva = load_vote_actors(args.model)      # vote appris, si le modèle en contient un
        if cva is not None and args.crew_ai == "mappo":
            crew_vote = VoteCtrl(cva, seed=3, greedy=args.greedy)
        if iva is not None and args.imp_ai == "mappo":
            imp_vote = VoteCtrl(iva, seed=4, greedy=args.greedy)
    crew = {"mappo": lambda: PolicyCtrl(crew_a, seed=1, greedy=args.greedy, name="Crew MAPPO"),
            "scripted": ScriptedCrew, "random": lambda: RandomCtrl(1)}[args.crew_ai]()
    imp = {"mappo": lambda: PolicyCtrl(imp_a, seed=2, greedy=args.greedy, name="Imposteur MAPPO"),
           "scripted": lambda: ScriptedImposter(2), "prudent": lambda: ScriptedImposter(2, prudent=True),
           "random": lambda: RandomCtrl(2)}[args.imp_ai]()

    from amongus.rt_gui import RTApp
    RTApp(Config(), crew, imp, human=None if args.spectate else 0, human_role=args.role, seed=args.seed,
          god=args.god, speed=args.speed, crew_vote=crew_vote, imp_vote=imp_vote).run()


if __name__ == "__main__":
    main()
