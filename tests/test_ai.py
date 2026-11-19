import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amongus.ai import obs as O
from amongus.ai.bots import RandomCtrl, ScriptedCrew, ScriptedImposter
from amongus.ai.policy import NumpyActor, PolicyCtrl, load_actors, save_actors
from amongus.env import N_ACTIONS
from amongus.config import Config
from amongus.rt_world import RTWorld, run_game


def random_actor(d_in, h=16, seed=0):
    r = np.random.default_rng(seed)
    return [(r.normal(0, .3, (d_in, h)), np.zeros(h)), (r.normal(0, .3, (h, N_ACTIONS)), np.zeros(N_ACTIONS))]


class TestAI(unittest.TestCase):
    def test_obs_shapes_and_finite(self):
        oc, oi, cc, ci = O.dims()
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=3)
        run_game(w, ScriptedCrew(), ScriptedImposter(1))
        w.reset(imposter=2)
        w.set_controllers(ScriptedCrew(), ScriptedImposter(1))
        for _ in range(300):
            w.update(0.1)
        for p in range(w.n):
            if not w.alive[p]:
                continue
            o, c, m = O.build(w, p, w.legal_actions(p))
            self.assertEqual(len(o), oi if p == w.imp else oc)
            self.assertEqual(len(c), ci if p == w.imp else cc)
            self.assertTrue(np.isfinite(o).all() and np.isfinite(c).all())
            self.assertTrue(m.any())

    def test_policy_picks_legal_action_and_records(self):
        oc, oi, cc, ci = O.dims()
        crew = PolicyCtrl(NumpyActor(random_actor(oc)), seed=1)
        imp = PolicyCtrl(NumpyActor(random_actor(oi)), seed=2)
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=4)
        run_game(w, crew, imp, record_crew=True, record_imp=True)
        tr = [t for p in range(w.n) for t in [w.traj[p]] if t]
        self.assertTrue(tr)
        for t in tr:
            self.assertTrue(t[-1]["done"])
            for s in t:
                self.assertTrue(s["mask"][s["act"]])
                self.assertLessEqual(s["logp"], 0.0)
                self.assertGreater(s["dur"], 0)

    def test_save_load_roundtrip(self):
        oc, oi, *_ = O.dims()
        a, b = random_actor(oc), random_actor(oi, seed=5)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "m.npz")
            save_actors(f, a, b)
            ca, ia = load_actors(f)
        x = np.random.default_rng(0).normal(size=oc).astype(np.float32)
        np.testing.assert_allclose(ca.logits(x), NumpyActor(a).logits(x), rtol=1e-5)

    def test_bots_finish_games(self):
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=9)
        for imp in (ScriptedImposter(1), ScriptedImposter(1, prudent=True), RandomCtrl(1)):
            r = run_game(w, ScriptedCrew(), imp)
            self.assertIn(r["winner"], ("crew", "imp"))

    def test_ppo_update_runs(self):
        try:
            import torch  # noqa
        except ImportError:
            self.skipTest("torch absent")
        from amongus.ai.mappo import RoleLearner
        from amongus.ai.rollout import _stack
        oc, oi, cc, ci = O.dims()
        crew = PolicyCtrl(NumpyActor(random_actor(oc)), seed=1)
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=4)
        run_game(w, crew, ScriptedImposter(1), record_crew=True)
        trajs = [_stack(w.traj[p]) for p in range(w.n) if w.traj[p]]
        learner = RoleLearner(oc, cc, hidden=16)
        st = learner.update(trajs, epochs=1, mb=64)
        self.assertGreater(st["n"], 0)
        self.assertTrue(np.isfinite(st["vl"]))

    def test_league_keeps_old_snapshots(self):
        from train_mappo import add_snap
        snaps = []
        for it in range(10, 1001, 10):
            snaps = add_snap(snaps, dict(act=it), it, every=10)
        its = [s["it"] for s in snaps]
        self.assertEqual(len(its), 24)
        self.assertEqual(its[-8:], list(range(930, 1001, 10)))       # les 8 plus récents
        self.assertTrue(all(i % 50 == 0 for i in its[:-8]))           # plus un ancien sur cinq
        self.assertLess(its[0], 300)                                  # dont certains très anciens


if __name__ == "__main__":
    unittest.main()
