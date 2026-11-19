"""Deux imposteurs (configuration par défaut : 5 crewmates + 2 imposteurs) et nombre de joueurs variable."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amongus import rt_map as M
from amongus.ai import claim as C
from amongus.ai import obs as O
from amongus.ai import vote as V
from amongus.ai.bots import ScriptedCrew, ScriptedImposter
from amongus.ai.policy import NumpyActor, PolicyCtrl
from amongus.ai.vote import VoteCtrl
from amongus.config import Config
from amongus.env import KILL
from amongus.rt_world import RTWorld, run_game


def actor(i, o, h=16, seed=0):
    r = np.random.default_rng(seed)
    return NumpyActor([(r.normal(0, 0.3, (i, h)).astype(np.float32), np.zeros(h, np.float32)),
                       (r.normal(0, 0.3, (h, o)).astype(np.float32), np.zeros(o, np.float32))])


def world(imps=(0, 1), **kw):
    w = RTWorld(Config(recall=1.0, **kw), seed=1)
    w.reset(imposter=list(imps))
    w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
    return w


class TestTwoImposters(unittest.TestCase):
    def test_default_config(self):
        cfg = Config()
        self.assertEqual((cfg.n_crew, cfg.n_imp, cfg.crew_left_to_lose), (5, 2, 0))
        w = RTWorld(seed=0)
        self.assertEqual((w.n, len(w.imps)), (7, 2))
        self.assertEqual(sum(1 for p in range(w.n) if w.tasks[p]), 5)        # seuls les crewmates ont des tâches
        self.assertEqual(Config.players("6+2").n_crew, 6)

    def test_parity_and_ejection(self):
        w = world()
        self.assertIsNone(w._outcome())
        w.alive[2] = w.alive[3] = False                 # 3 crewmates contre 2 imposteurs : on continue
        self.assertIsNone(w._outcome())
        w.alive[4] = False                              # 2 contre 2 : les imposteurs gagnent
        self.assertEqual(w._outcome(), ("imp", "parity"))
        w = world()
        w.alive[0] = False                              # un imposteur éjecté : la partie continue
        self.assertIsNone(w._outcome())
        w.alive[1] = False
        self.assertEqual(w._outcome(), ("crew", "ejected"))

    def test_imposters_do_not_kill_each_other(self):
        w = world()
        for q in range(w.n):
            w.x[q], w.y[q] = M.room_center(6)
        w._update_rooms()
        w.kcd = [0.0] * w.n
        self.assertIn(KILL, w.legal_actions(0))
        for q in range(2, w.n):                         # seuls les imposteurs restent dans la salle
            w.x[q], w.y[q] = M.task_stand(5)
        w._update_rooms()
        self.assertNotIn(KILL, w.legal_actions(0))     # le partenaire n'est pas une cible

    def test_cooldowns_are_per_imposter(self):
        w = world()
        w.kcd = [0.0] * w.n
        w.x[0], w.y[0] = w.x[2] + 20, w.y[2]
        w._update_rooms()
        w.do_kill(0, 2)
        self.assertGreater(w.kcd[0], 0)
        self.assertEqual(w.kcd[1], 0.0)
        self.assertEqual(w.kills[-1].killer, 0)

    def test_witness_accuses_the_right_killer(self):
        w = world()
        for q in range(w.n):
            w.x[q], w.y[q] = M.task_stand(5)
        for q in (1, 2, 3):                             # 1 tue 2 sous les yeux de 3 ; 0 est loin
            w.x[q], w.y[q] = M.room_center(6)
        w.x[3] += 60
        w._update_rooms()
        w.time = 20.0
        w._observe()
        w.do_kill(1, 2)
        w.vote_callbacks = {p: (lambda env, v, ctx: None) for p in range(w.n)}
        w.start_meeting(3)
        ctx = [e for e in w.events if e["type"] == "meeting"][-1]["ctx"]
        self.assertEqual(ctx.accuse.get(3), 1)
        self.assertIn(0, ctx.claims)                    # les deux imposteurs déclarent
        self.assertIn(1, ctx.claims)
        self.assertNotIn(ctx.accuse.get(0), (0, 1))     # un imposteur n'accuse pas son partenaire

    def test_partner_flags_and_masks(self):
        w = world()
        o = O.local_obs(w, 0)
        oth = o[-4 - O.N_OTHERS * O.OTHER_F:-4].reshape(O.N_OTHERS, O.OTHER_F)
        self.assertEqual(oth[0, 10], 1.0)               # case 1 = joueur 1 : partenaire
        self.assertEqual(oth[1, 10], 0.0)
        self.assertEqual(oth[w.n - 1, 9], 0.0)          # 7 joueurs : les cases 7 à 9 sont vides
        oc = O.local_obs(w, 3).reshape(-1)
        crew_rows = oc[-4 - O.N_OTHERS * O.OTHER_F:-4].reshape(O.N_OTHERS, O.OTHER_F)
        self.assertFalse(crew_rows[:, 10].any())        # un crewmate ne sait pas qui est imposteur
        m = V.vote_mask(w, 0)
        self.assertFalse(m[0])                          # pas de vote contre son partenaire
        cm = C.claim_mask(w, 0).reshape(-1, C.N_ACC)
        self.assertFalse(cm[:, 0].any())                # ni d'accusation

    def test_same_network_sizes_for_7_8_and_10_players(self):
        sizes = []
        for crew, imp in ((5, 2), (6, 2), (8, 2), (6, 1)):
            w = RTWorld(Config(n_crew=crew, n_imp=imp), seed=0)
            p, q = w.imps[0], next(x for x in range(w.n) if x not in w.imps)
            from amongus.env import MeetingCtx
            ctx = MeetingCtx(reporter=q, bodies=[], cleared=set(), witnesses=[], imposter=p)
            sizes.append((len(O.local_obs(w, q)), len(O.local_obs(w, p)), len(O.global_state(w, p)),
                          len(V.vote_obs(w, q, ctx)), len(C.claim_obs(w, p, ctx))))
        self.assertEqual(len(set(sizes)), 1, sizes)

    def test_full_games_with_networks(self):
        oc, oi, _, _ = O.dims()
        vo, _ = V.dims()
        co, _ = C.dims()
        for crew, imp in ((5, 2), (6, 2)):
            w = RTWorld(Config(n_crew=crew, n_imp=imp), seed=3)
            r = run_game(w, PolicyCtrl(actor(oc, 21), seed=1), PolicyCtrl(actor(oi, 21, seed=1), seed=2),
                         record_crew=True, record_imp=True, crew_vote=VoteCtrl(actor(vo, V.N_VOTE, seed=2), seed=3),
                         imp_vote=VoteCtrl(actor(vo, V.N_VOTE, seed=3), seed=4),
                         imp_claim=C.ClaimCtrl(actor(co, C.N_CLAIM, seed=4), seed=5))
            self.assertIn(r["winner"], ("crew", "imp"))
            self.assertTrue(any(w.traj[i] for i in w.imps))
            for i in w.imps:                            # les deux imposteurs ont des trajectoires
                for s in w.traj[i]:
                    self.assertTrue(s["mask"][s["act"]])


if __name__ == "__main__":
    unittest.main()
