"""Conduits d'aération et sabotage des lumières."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amongus import rt_map as M
from amongus.ai import obs as O
from amongus.ai.bots import RandomCtrl, ScriptedCrew, ScriptedImposter
from amongus.config import Config
from amongus.env import FIX, SABOTAGE, VENT_A, VENT_B
from amongus.rt_world import RTWorld, run_game


def world(imposter=0, recall=1.0):
    w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2, recall=recall), seed=1)
    w.reset(imposter=imposter)
    w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
    return w


def put(w, p, pos):
    w.x[p], w.y[p] = pos
    w._update_rooms()


def run_macro(w, p, macro, max_s=20.0):
    w.ag[p]["macro"], w.ag[p]["t0"] = macro, w.time
    t = 0.0
    while w.ag[p]["macro"] is not None and t < max_s:
        w.time += 0.1
        w._ai_tick(p, 0.1)
        t += 0.1


class TestVents(unittest.TestCase):
    def test_vent_map(self):
        for r, links in M.VENT_LINKS.items():
            self.assertEqual(len(links), 2)
            x, y = M.vent_pos(r)
            self.assertTrue(M.can_stand(x, y))
            self.assertEqual(M.room_at(x, y), r)
            for t in links:
                self.assertIn(r, M.VENT_LINKS[t])
        self.assertTrue(M.can_stand(*M.fix_pos()))
        self.assertEqual(M.room_at(*M.fix_pos()), M.FIX_ROOM)

    def test_vent_actions_only_for_imposter_in_vent_room(self):
        w = world()
        put(w, 0, M.room_center(1))
        put(w, 2, M.room_center(1))
        self.assertTrue({VENT_A, VENT_B} <= set(w.legal_actions(0)))
        self.assertFalse({VENT_A, VENT_B} & set(w.legal_actions(2)))
        put(w, 0, M.room_center(0))                    # pas de grille en Cafétéria
        self.assertFalse({VENT_A, VENT_B} & set(w.legal_actions(0)))

    def test_unseen_vent_travel(self):
        w = world()
        for q in range(1, w.n):                        # tout le monde en Cafétéria, loin
            put(w, q, M.room_center(0))
        put(w, 0, M.room_center(1))
        run_macro(w, 0, ("vent", 1, 4))
        self.assertEqual(w.room[0], 4)
        self.assertFalse(w.in_vent)
        self.assertEqual(w.vent_seen.get(0, []), [])
        self.assertTrue(any(e["type"] == "vent" for e in w.events))

    def test_seen_venting_is_direct_evidence(self):
        w = world()
        for q in range(1, w.n):
            put(w, q, M.room_center(0))
        put(w, 0, M.room_center(1))
        put(w, 3, M.room_center(1))                    # 3 regarde
        run_macro(w, 0, ("vent", 1, 4))
        self.assertIn(3, w.vent_seen[0])
        self.assertIn(0, w.saw_kill[3])
        w.vote_callbacks = {p: (lambda env, v, ctx: None) for p in range(w.n)}
        w.start_meeting(3, emergency=True)
        ctx = [e for e in w.events if e["type"] == "meeting"][-1]["ctx"]
        self.assertIn(3, ctx.witnesses)
        self.assertEqual(w.vent_seen, {})              # remis à zéro après la réunion

    def test_invisible_inside_vent(self):
        w = world()
        put(w, 0, M.room_center(1))
        put(w, 2, M.room_center(4))
        w.vent(0, 4)
        self.assertTrue(w.in_vent)
        self.assertFalse(w.sees(2, 0))
        w._vent_exit(0)
        self.assertTrue(w.sees(2, 0))
        self.assertIn(2, w.vent_seen[0])                  # il l'a vu sortir


class TestLights(unittest.TestCase):
    def test_sabotage_reduces_crew_vision_only(self):
        w = world()
        put(w, 0, M.room_center(6))
        put(w, 2, M.room_center(6))
        w.x[2] += 250
        self.assertTrue(w.sees(2, 0))
        self.assertNotIn(SABOTAGE, w.legal_actions(0))  # recharge du début de partie
        w.time = w.rt.sabotage_cooldown + 1
        self.assertIn(SABOTAGE, w.legal_actions(0))
        self.assertTrue(w.sabotage())
        self.assertFalse(w.sees(2, 0))                  # même salle, mais trop loin dans le noir
        self.assertTrue(w.sees(0, 2))                   # l'imposteur voit toujours
        self.assertNotIn(SABOTAGE, w.legal_actions(0))

    def test_fix_in_electrical(self):
        w = world()
        w.time = w.rt.sabotage_cooldown + 1
        w.sabotage()
        put(w, 2, M.room_center(M.FIX_ROOM))
        put(w, 3, M.room_center(0))
        self.assertIn(FIX, w.legal_actions(2))
        self.assertNotIn(FIX, w.legal_actions(3))
        run_macro(w, 2, ("fix",))
        self.assertFalse(w.lights_off)
        self.assertGreater(w.sab_ready_t, w.time)
        ev = [e for e in w.events if e["type"] == "lights"]
        self.assertEqual((ev[-1]["on"], ev[-1]["fixer"]), (True, 2))

    def test_lights_come_back_alone_and_after_meeting(self):
        w = world()
        w.time = w.rt.sabotage_cooldown + 1
        w.sabotage()
        w.time += w.rt.lights_max
        w.update(0.1)
        self.assertFalse(w.lights_off)
        w.time = w.sab_ready_t + 1
        w.sabotage()
        w.vote_callbacks = {p: (lambda env, v, ctx: None) for p in range(w.n)}
        w.start_meeting(2, emergency=True)
        self.assertFalse(w.lights_off)

    def test_kill_in_the_dark_has_fewer_witnesses(self):
        def kill(dark):
            w = world()
            put(w, 0, M.room_center(6))
            put(w, 1, M.room_center(6))
            w.x[1] += 30
            put(w, 2, M.room_center(6))
            w.x[2] += 260                               # même salle, à 260 px
            if dark:
                w.time = w.rt.sabotage_cooldown + 1
                w.sabotage()
            w.do_kill(0, 1)
            return w.kills[-1]
        self.assertEqual(kill(False).witnesses, [2])
        k = kill(True)
        self.assertEqual(k.witnesses, [])
        self.assertTrue(k.dark)

    def test_observation_signals(self):
        w = world()
        o0 = O.local_obs(w, 2)
        w.time = w.rt.sabotage_cooldown + 1
        w.sabotage()
        put(w, 2, M.room_center(M.FIX_ROOM))
        o1 = O.local_obs(w, 2)
        self.assertEqual((o0[9], o1[9]), (0.0, 1.0))    # lumières coupées
        self.assertEqual(o1[12], 1.0)                   # peut réparer ici
        self.assertEqual(len(o0), len(o1))

    def test_full_games_still_run(self):
        for seed in range(3):
            w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=seed)
            r = run_game(w, RandomCtrl(seed), RandomCtrl(seed + 1))
            self.assertIn(r["winner"], ("crew", "imp"))


if __name__ == "__main__":
    unittest.main()
