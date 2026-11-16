import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amongus import rt_map as M
from amongus.ai.bots import ScriptedCrew, ScriptedImposter
from amongus.config import Config
from amongus.rt_world import RTWorld, run_game


class TestRT(unittest.TestCase):
    def test_topology(self):
        self.assertEqual(M.components_topology(), M.EDGE_SET)

    def test_collisions_and_spawn(self):
        for x, y in M.spawn_points(7):
            self.assertTrue(M.can_stand(x, y))
        self.assertFalse(M.can_stand(5, 5))
        self.assertEqual(M.room_at(*M.room_center(3)), 3)

    def test_paths(self):
        for r in range(12):
            f = M.field(("room", r))
            for s in range(12):
                cx, cy = M.room_center(s)
                self.assertLess(f[int(cy) // M.NAV * M.NW + int(cx) // M.NAV], 1e9)

    def test_full_game_and_determinism(self):
        res = []
        for _ in range(2):
            w = RTWorld(Config(), seed=3)
            res.append(run_game(w, ScriptedCrew(), ScriptedImposter(2)))
        self.assertIn(res[0]["winner"], ("crew", "imp"))
        self.assertEqual(res[0], res[1])

    def test_kill_and_meeting(self):
        w = RTWorld(Config(), seed=1)
        w.reset(0)
        w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
        v = 1
        w.x[0], w.y[0] = w.x[v] + 30, w.y[v]
        w.cd_s = 0
        n = len(w.crew_alive())
        w.do_kill(0, v)
        self.assertFalse(w.alive[v])
        self.assertEqual(len(w.crew_alive()), n - 1)
        w.update(0.1)
        ev = w.drain_events()
        self.assertTrue(any(e["type"] == "kill" for e in ev))

    def test_imposter_can_report(self):
        """Comme dans le vrai jeu, l'imposteur peut signaler un corps : celui qui signale n'est pas forcément innocent."""
        from amongus.env import REPORT
        w = RTWorld(Config(), seed=1)
        w.reset(0)
        w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
        self.assertNotIn(REPORT, w.legal_actions(0))            # pas de corps : rien à signaler
        v = 1
        for q in range(2, w.n):                                 # les autres sont loin : pas de signalement automatique
            w.x[q], w.y[q] = M.task_stand(5)
        w.x[0], w.y[0] = w.x[v] + 30, w.y[v]
        w._update_rooms()
        w.do_kill(0, v)
        self.assertIn(REPORT, w.legal_actions(0))
        self.assertNotIn(REPORT, w.legal_actions(2))            # réservé à l'imposteur (les crewmates IA signalent seuls)
        w.ag[0]["macro"] = ("report",)
        w._ai_tick(0, 0.1)
        ev = [e for e in w.events if e["type"] == "meeting"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["reporter"], 0)

    def test_old_model_without_report_action(self):
        """Un acteur à 15 sorties (avant « signaler ») reste jouable : il ne signale simplement jamais."""
        import numpy as np
        from amongus.ai import obs as O
        from amongus.ai.policy import NumpyActor, PolicyCtrl
        from amongus.env import N_ACTIONS, REPORT
        oc, oi, _, _ = O.dims()
        r = np.random.default_rng(0)
        old = lambda d: [(r.normal(0, .3, (d, 8)), np.zeros(8)), (r.normal(0, .3, (8, N_ACTIONS - 1)), np.zeros(N_ACTIONS - 1))]
        w = RTWorld(Config(), seed=2)
        res = run_game(w, PolicyCtrl(NumpyActor(old(oc)), seed=1), PolicyCtrl(NumpyActor(old(oi)), seed=2),
                       record_imp=True)
        self.assertIn(res["winner"], ("crew", "imp"))
        self.assertTrue(all(s["act"] != REPORT for s in w.traj[w.imp]))


if __name__ == "__main__":
    unittest.main()
