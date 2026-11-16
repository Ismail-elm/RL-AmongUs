"""Tests du moteur de règles.   Lancer :  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amongus.config import Config
from amongus.env import AmongUsEnv, KILL, TASK, WAIT, N_ROOMS, N_ACTIONS
from amongus.mapdata import ADJ, DIST, NEXT_HOP


def make_env(**kw):
    cfg = Config(**kw)
    env = AmongUsEnv(cfg, seed=1)
    env.reset(imposter=6)
    return env


class TestMap(unittest.TestCase):
    def test_connected_and_symmetric(self):
        for a in range(N_ROOMS):
            for b in ADJ[a]:
                self.assertIn(a, ADJ[b])
            for b in range(N_ROOMS):
                self.assertGreaterEqual(DIST[a][b], 0)

    def test_next_hop_gets_closer(self):
        for a in range(N_ROOMS):
            for b in range(N_ROOMS):
                if a != b:
                    self.assertEqual(DIST[NEXT_HOP[a][b]][b], DIST[a][b] - 1)


class TestRules(unittest.TestCase):
    def test_start(self):
        env = make_env()
        self.assertEqual(env.n, 7)
        self.assertTrue(all(r == 0 for r in env.room))
        self.assertEqual(len(env.tasks[6]), 0)  # l'imposteur n'a pas de tâches
        self.assertTrue(all(len(env.tasks[p]) == env.cfg.tasks_per_crew for p in range(6)))
        self.assertNotIn(0, env.tasks[0])

    def test_illegal_action_becomes_wait(self):
        env = make_env()
        env.step({0: 5})  # Stockage est voisine de la Cafétéria (0-5), mais 9 ne l'est pas
        self.assertEqual(env.room[0], 5)
        env.step({0: 9})
        self.assertEqual(env.room[0], 5)

    def test_task_takes_duration_and_resets_on_interruption(self):
        env = make_env(task_duration=3)
        env.room[0] = 3
        env.tasks[0] = {3}
        env.step({0: TASK})
        env.step({0: TASK})
        self.assertEqual(env.tasks[0], {3})
        env.step({0: WAIT})              # interruption
        self.assertEqual(env.progress[0], 0)
        for _ in range(3):
            rew, _, _ = env.step({0: TASK})
        self.assertEqual(env.tasks[0], set())
        self.assertGreater(rew[0], env.cfg.r_task - 1)

    def test_kill_needs_cooldown_and_presence(self):
        env = make_env(kill_cooldown=2)
        self.assertNotIn(KILL, env.legal_actions(6))   # cooldown initial
        env.step({}); env.step({})
        self.assertIn(KILL, env.legal_actions(6))      # prêt + crewmates présents
        env.room[6] = 3                                # imposteur seul
        self.assertNotIn(KILL, env.legal_actions(6))

    def test_witnessed_kill_gets_imposter_ejected(self):
        env = make_env(kill_cooldown=0, skip_threshold=3)
        for p in range(7):
            env.room[p] = 4 if p in (6, 0, 1) else 1 + p % 3  # imposteur + 2 crewmates ensemble
        rew, term, done = env.step({6: KILL})
        kills = [e for e in env.events if e["type"] == "kill"]
        self.assertEqual(len(kills), 1)
        meeting = [e for e in env.events if e["type"] == "meeting"][0]
        self.assertTrue(len(meeting["witnesses"]) == 1)
        self.assertEqual(meeting["ejected"], 6)        # le témoin identifie l'imposteur
        self.assertEqual(env.winner, "crew")
        self.assertEqual(env.reason, "ejected")
        self.assertTrue(term[6])
        self.assertLess(rew[6], 0)

    def test_unwitnessed_kill_triggers_meeting_only_when_body_found(self):
        env = make_env(kill_cooldown=0)
        for p in range(7):
            env.room[p] = 8 if p in (6, 0) else 1 + p % 3
        env.step({6: KILL})
        self.assertFalse(env.alive[0])
        self.assertEqual(env.bodies, [(8, 0)])
        self.assertFalse([e for e in env.events if e["type"] == "meeting"])
        env.imp_mem = None
        env.room[1] = 7
        env.step({1: 8})                              # un crewmate arrive sur le corps
        self.assertTrue([e for e in env.events if e["type"] == "meeting"])
        self.assertEqual(env.bodies, [])
        self.assertTrue(all(env.room[p] == 0 for p in env.alive_players()))

    def test_crew_wins_by_tasks(self):
        env = make_env()
        for p in range(6):
            env.tasks[p] = set()
        env.step({})
        self.assertEqual((env.winner, env.reason), ("crew", "tasks"))

    def test_imposter_wins_by_parity_and_timeout(self):
        env = make_env(crew_left_to_lose=3)
        for p in (0, 1, 2):
            env.alive[p] = False
        env.step({})
        self.assertEqual((env.winner, env.reason), ("imp", "parity"))
        env = make_env(max_steps=2)
        env.step({}); env.step({})
        self.assertEqual((env.winner, env.reason), ("imp", "timeout"))

    def test_deterministic_with_seed(self):
        def play(seed):
            import random
            env = AmongUsEnv(Config(), seed=seed)
            rng = random.Random(seed + 1)
            while env.winner is None:
                env.step({q: rng.choice(env.legal_actions(q)) for q in env.alive_players()})
            return env.winner, env.reason, env.t, dict(env.stats)
        self.assertEqual(play(5), play(5))



if __name__ == "__main__":
    unittest.main()
