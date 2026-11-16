import os
import tempfile
import unittest

import numpy as np

from amongus.ai import obs as O
from amongus.ai import vote as V
from amongus.ai.bots import ScriptedCrew, ScriptedImposter
from amongus.ai.policy import NumpyActor, PolicyCtrl, load_actors, load_vote_actors, save_actors
from amongus.env import MeetingCtx, N_ACTIONS
from amongus.rt_world import RTWorld, run_game


def random_actor(i, o, h=16, seed=0):
    r = np.random.default_rng(seed)
    return [(r.normal(0, 0.3, (i, h)).astype(np.float32), np.zeros(h, np.float32)),
            (r.normal(0, 0.3, (h, o)).astype(np.float32), np.zeros(o, np.float32))]


class FixedVote:
    """Vote toujours contre le joueur `target` (pour tester l'éjection)."""

    def __init__(self, target):
        self.target = target

    def vote(self, world, voter, ctx, record=False):
        return (None if voter == self.target else self.target), None


def play_until_meeting(crew_vote=None, imp_vote=None, record=False, max_seeds=30):
    """Joue des parties jusqu'à en trouver une avec au moins une réunion."""
    oc, oi, _, _ = O.dims()
    for seed in range(max_seeds):
        w = RTWorld(seed=seed)
        r = run_game(w, PolicyCtrl(NumpyActor(random_actor(oc, N_ACTIONS)), seed=seed) if record else ScriptedCrew(),
                     ScriptedImposter(seed), record_crew=record, record_imp=False,
                     crew_vote=crew_vote, imp_vote=imp_vote)
        if r["meetings"] > 0:
            return w, r
    raise AssertionError("aucune réunion en %d parties" % max_seeds)


class TestVote(unittest.TestCase):
    def test_obs_and_mask(self):
        vo, vc = V.dims()
        w = RTWorld(seed=1)
        w.reset(imposter=3)
        w.alive[5] = False
        ctx = MeetingCtx(reporter=2, bodies=[(1, 5)], cleared={4}, witnesses=[], imposter=3)
        o, c, m = V.build(w, 0, ctx)
        self.assertEqual((len(o), len(c)), (vo, vc))
        self.assertTrue(np.isfinite(o).all())
        self.assertEqual(m.tolist(), [True, True, True, True, False, True, True])   # joueur 5 mort, « passer » permis

    def test_action_maps_to_player(self):
        vo, _ = V.dims()
        w = RTWorld(seed=1)
        w.reset(imposter=3)
        ctx = MeetingCtx(reporter=2, bodies=[], cleared=set(), witnesses=[], imposter=3)
        ctrl = V.VoteCtrl(NumpyActor(random_actor(vo, V.N_VOTE)), seed=0)
        for voter in range(w.n):
            for _ in range(20):
                choice, rec = ctrl.vote(w, voter, ctx, record=True)
                self.assertNotEqual(choice, voter)
                if rec["act"] == V.SKIP:
                    self.assertIsNone(choice)
                else:
                    self.assertEqual(choice, (voter + rec["act"] + 1) % w.n)

    def test_default_is_heuristic(self):
        """Sans réseau de vote, le comportement est celui de la règle de base (parties identiques)."""
        a = run_game(RTWorld(seed=5), ScriptedCrew(), ScriptedImposter(2))
        b = run_game(RTWorld(seed=5), ScriptedCrew(), ScriptedImposter(2), crew_vote=None, imp_vote=None)
        self.assertEqual(a, b)

    def test_unanimous_vote_ejects(self):
        """Si tous les crewmates votent contre le même joueur, il est éjecté à la première réunion."""
        for seed in range(30):
            w = RTWorld(seed=seed)
            w.reset()
            imp = w.imp
            w = RTWorld(seed=seed)
            r = run_game(w, ScriptedCrew(), ScriptedImposter(seed), imposter=imp, crew_vote=FixedVote(imp))
            if r["meetings"] > 0:
                self.assertEqual((r["winner"], r["reason"], r["good_ejections"]), ("crew", "ejected", 1))
                return
        self.fail("aucune réunion")

    def test_records_vote_trajectories(self):
        vo, vc = V.dims()
        ctrl = V.VoteCtrl(NumpyActor(random_actor(vo, V.N_VOTE)), seed=1)
        w, r = play_until_meeting(crew_vote=ctrl, record=True)
        trajs = [w.vtraj[p] for p in range(w.n) if w.vtraj[p]]
        self.assertTrue(trajs)
        self.assertFalse(w.vtraj[w.imp])                      # l'imposteur n'est pas enregistré ici
        self.assertEqual(max(len(t) for t in trajs), r["meetings"])
        for t in trajs:
            for s in t:
                self.assertTrue(s["done"])                    # chaque vote est un épisode à lui seul
                self.assertEqual(len(s["obs"]), vo)
                self.assertEqual(len(s["crit"]), vc)
                self.assertTrue(s["mask"][s["act"]])
                self.assertGreater(s["dur"], 0)

    def test_vote_rewards(self):
        """Voter contre l'imposteur et l'éjecter : +1 (vote juste) +5 (éjection) pour chaque crewmate votant."""
        vo, _ = V.dims()

        class Accuse(V.VoteCtrl):
            def vote(self, world, voter, ctx, record=False):
                _, rec = super().vote(world, voter, ctx, record)
                return world.imp, rec

        ctrl = Accuse(NumpyActor(random_actor(vo, V.N_VOTE)), seed=1)
        w, r = play_until_meeting(crew_vote=ctrl, record=True)
        self.assertEqual(r["reason"], "ejected")
        recs = [s for p in range(w.n) for s in w.vtraj[p]]
        self.assertTrue(recs)
        self.assertTrue(all(abs(s["r"] - (w.rw.vote_right + w.rw.vote_eject_imp)) < 1e-6 for s in recs))

    def test_save_load_with_votes(self):
        oc, oi, _, _ = O.dims()
        vo, _ = V.dims()
        a, b = random_actor(oc, N_ACTIONS), random_actor(oi, N_ACTIONS)
        va, vb = random_actor(vo, V.N_VOTE, seed=1), random_actor(vo, V.N_VOTE, seed=2)
        with tempfile.TemporaryDirectory() as d:
            f, g = os.path.join(d, "m.npz"), os.path.join(d, "old.npz")
            save_actors(f, a, b, crew_vote=va, imp_vote=vb)
            save_actors(g, a, b)
            ca, ia = load_actors(f)
            cv, iv = load_vote_actors(f)
            self.assertEqual(load_vote_actors(g), (None, None))      # ancien format : pas de vote appris
        self.assertEqual(len(ca.params), 2)
        x = np.random.default_rng(0).normal(size=vo).astype(np.float32)
        np.testing.assert_allclose(cv.logits(x), NumpyActor(va).logits(x), rtol=1e-5)
        np.testing.assert_allclose(iv.logits(x), NumpyActor(vb).logits(x), rtol=1e-5)

    def test_teacher_examples_and_imitation(self):
        """Le pré-entraînement : la règle heuristique vote, et chaque crewmate fournit un exemple étiqueté."""
        try:
            from amongus.ai.mappo import RoleLearner
        except ImportError:
            self.skipTest("PyTorch non installé")
        from amongus.ai.rollout import _stack
        vo, vc = V.dims()
        w, _ = play_until_meeting(crew_vote=V.TeacherVote(), record=True)
        self.assertFalse(w.vtraj[w.imp])
        for p in range(w.n):
            for s in w.vtraj[p]:
                self.assertEqual((p + s["act"] + 1) % w.n, w.imp)      # l'étiquette désigne bien l'imposteur
        trajs = [_stack(w.vtraj[p]) for p in range(w.n) if w.vtraj[p]]
        acc, n = RoleLearner(vo, vc, hidden=16, n_actions=V.N_VOTE).imitate(trajs, epochs=2)
        self.assertGreater(n, 0)
        self.assertTrue(0.0 <= acc <= 1.0)

    def test_learner_update_on_votes(self):
        try:
            from amongus.ai.mappo import RoleLearner
        except ImportError:
            self.skipTest("PyTorch non installé")
        from amongus.ai.rollout import _stack
        vo, vc = V.dims()
        ctrl = V.VoteCtrl(NumpyActor(random_actor(vo, V.N_VOTE)), seed=1)
        w, _ = play_until_meeting(crew_vote=ctrl, record=True)
        trajs = [_stack(w.vtraj[p]) for p in range(w.n) if w.vtraj[p]]
        st = RoleLearner(vo, vc, hidden=16, n_actions=V.N_VOTE).update(trajs, epochs=1, mb=64)
        self.assertGreater(st["n"], 0)


if __name__ == "__main__":
    unittest.main()
