"""Mesures ciblées (IA contre IA) : réunions d'urgence et fins de partie décisives.

    python tools_measure.py [modèle.npz] [nb_parties]
"""
import multiprocessing as mp
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Spy:
    """Enveloppe un contrôleur d'imposteur : note ses choix quand un meurtre ferait gagner tout de suite."""

    def __init__(self, ctrl, stats):
        self.ctrl, self.stats = ctrl, stats
        self.name = getattr(ctrl, "name", "spy")

    def decide(self, world, p, legal, record=False):
        from amongus.env import KILL, VENT_A, VENT_B
        a, rec = self.ctrl.decide(world, p, legal, record)
        crew, imps = len(world.crew_alive()), len(world.imps_alive())
        if crew - 1 <= imps and imps > 0:                      # un meurtre de plus = victoire
            self.stats["decisive_decisions"] += 1
            if KILL in legal:
                self.stats["decisive_kill_legal"] += 1
                self.stats["decisive_kill_chosen"] += int(a == KILL)
                self.stats["decisive_vent_chosen_when_kill"] += int(a in (VENT_A, VENT_B))
        return a, rec

    def __getattr__(self, k):
        return getattr(self.ctrl, k)


def play(args):
    model, seeds = args
    from amongus.ai.claim import ClaimCtrl
    from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
    from amongus.ai.vote import VoteCtrl
    from amongus.config import Config
    from amongus.rt_world import RTWorld
    ca, ia = load_actors(model)
    cv, iv = load_vote_actors(model)
    ic = load_claim_actor(model)
    st = Counter()
    em = []          # (appelant imposteur ?, temps, résultat)
    ends = []        # fins décisives : (durée de l'état décisif, issue)
    for s in seeds:
        w = RTWorld(Config.players(), seed=s)
        w.reset()
        spy = Spy(PolicyCtrl(ia, seed=s + 2), st)
        w.set_controllers(PolicyCtrl(ca, seed=s + 1), spy, crew_vote=VoteCtrl(cv, seed=s + 3),
                          imp_vote=VoteCtrl(iv, seed=s + 4), imp_claim=ClaimCtrl(ic, seed=s + 5))
        dec_t0 = None
        while w.winner is None and w.time < 2000:
            crew, imps = len(w.crew_alive()), len(w.imps_alive())
            if dec_t0 is None and imps > 0 and crew - 1 <= imps:
                dec_t0 = (w.time, crew, imps)
            w.update(0.1)
            for e in w.drain_events():
                if e["type"] == "meeting":
                    who = "imp" if e["reporter"] in w.imps else "crew"
                    if e.get("emergency"):
                        res = "none" if e["ejected"] is None else ("imp" if e["was_imposter"] else "innocent")
                        em.append((who, round(w.time - w.rt.meeting_time), res))
                    st["meetings"] += 1
                    st["meet_" + ("emerg" if e.get("emergency") else "body")] += 1
                    if e["ejected"] is not None:
                        st["ej_imp" if e["was_imposter"] else "ej_inno"] += 1
            if dec_t0 is not None and (len(w.crew_alive()) - 1 > len(w.imps_alive()) or not w.imps_alive()):
                pass
        if dec_t0 is not None:
            ends.append((round(w.time - dec_t0[0], 1), f"{dec_t0[1]}v{dec_t0[2]}", w.winner, w.reason))
        st["games"] += 1
        st["win_" + (w.winner or "imp")] += 1
    return st, em, ends


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "models/mappo.npz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 440
    k = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(k) as pool:
        res = pool.map(play, [(model, list(range(i, n, k))) for i in range(k)])
    st, em, ends = Counter(), [], []
    for a, b, c in res:
        st.update(a)
        em += b
        ends += c
    g = st["games"]
    print(f"== {model} : {g} parties IA contre IA, crew gagne {100 * st['win_crew'] / g:.0f} %")
    print(f"réunions/partie {st['meetings'] / g:.2f} (urgences {100 * st['meet_emerg'] / max(1, st['meetings']):.0f} %), "
          f"imposteur éjecté {100 * st['ej_imp'] / max(1, st['meetings']):.0f} %, innocent éjecté "
          f"{100 * st['ej_inno'] / max(1, st['meetings']):.0f} %")
    if em:
        by = Counter(x[0] for x in em)
        print(f"urgences : {len(em) / g:.2f}/partie ; appelées par un imposteur {100 * by['imp'] / len(em):.0f} %")
        for who in ("imp", "crew"):
            sub = [x for x in em if x[0] == who]
            if sub:
                r = Counter(x[2] for x in sub)
                early = sum(1 for x in sub if x[1] < 40)
                print(f"  appelées par {who:4s} : {len(sub):4d}, avant 40 s : {100 * early / len(sub):.0f} %, résultat : "
                      + ", ".join(f"{k} {100 * v / len(sub):.0f} %" for k, v in r.most_common()))
        r = Counter(x[2] for x in em if x[1] < 40)
        tot = sum(r.values())
        if tot:
            print("  urgences avant 40 s : " + ", ".join(f"{k} {100 * v / tot:.0f} %" for k, v in r.most_common()))
    if ends:
        lone = [e for e in ends if e[1].endswith("v1")]
        print(f"fins décisives (un meurtre = victoire) : {len(ends)} parties ({100 * len(ends) / g:.0f} %), "
              f"dont imposteur seul : {len(lone)}")
        for name, sub in (("toutes", ends), ("imposteur seul", lone)):
            if sub:
                r = Counter((e[2], e[3]) for e in sub)
                print(f"  {name} : durée moyenne de l'état {sum(e[0] for e in sub) / len(sub):.0f} s ; issues : "
                      + ", ".join(f"{a}/{b} {100 * v / len(sub):.0f} %" for (a, b), v in r.most_common()))
    if st["decisive_kill_legal"]:
        print(f"décisions de l'imposteur en état décisif avec meurtre possible : {st['decisive_kill_legal']}, "
              f"il tue {100 * st['decisive_kill_chosen'] / st['decisive_kill_legal']:.0f} %, "
              f"prend un conduit {100 * st['decisive_vent_chosen_when_kill'] / st['decisive_kill_legal']:.0f} %")


if __name__ == "__main__":
    main()
