"""Règles proches du vrai jeu : morts connues seulement en réunion, déclarations (et mensonges) de l'imposteur,
recoupements par les crewmates, réunion d'urgence, victoire de l'imposteur à 1 crewmate."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amongus import rt_map as M
from amongus.ai import claim as C
from amongus.ai import obs as O
from amongus.ai import vote as V
from amongus.ai.bots import ScriptedCrew, ScriptedImposter
from amongus.ai.policy import NumpyActor, PolicyCtrl, load_actors, load_claim_actor, save_actors
from amongus.config import Config
from amongus.env import EMERGENCY, N_ACTIONS
from amongus.rt_world import RTWorld, run_game

FAR = 5          # salle loin de la Cafétéria (où sont les témoins / spectateurs)


def actor(i, o, h=16, seed=0):
    r = np.random.default_rng(seed)
    return [(r.normal(0, 0.3, (i, h)).astype(np.float32), np.zeros(h, np.float32)),
            (r.normal(0, 0.3, (h, o)).astype(np.float32), np.zeros(o, np.float32))]


def staged_kill(witness=None, imposter=0, victim=1, recall=1.0):
    """L'imposteur tue la victime en Cafétéria. Tous les autres sont loin (salle FAR), sauf `witness` qui reste à côté.
    recall = 1 : les crewmates se souviennent de tout (recoupements déterministes)."""
    w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2, recall=recall), seed=1)
    w.reset(imposter=imposter)
    w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
    for q in range(w.n):
        if q not in (imposter, victim, witness):
            w.x[q], w.y[q] = M.task_stand(FAR)
    w.x[imposter], w.y[imposter] = w.x[victim] + 30, w.y[victim]
    w._update_rooms()
    w.time = 20.0
    w._observe()
    w.do_kill(imposter, victim)
    return w


class TestRules(unittest.TestCase):
    def test_imposter_wins_at_two_crewmates(self):
        """Mode 1 imposteur avec le seuil historique : victoire quand il ne reste que 2 crewmates."""
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=0)
        w.reset(imposter=0)
        for v in range(1, w.n - 3):          # il reste 3 crewmates : la partie continue
            w.alive[v] = False
        self.assertIsNone(w._outcome())
        w.alive[w.n - 3] = False             # il n'en reste que 2 : l'imposteur gagne
        self.assertEqual(w._outcome(), ("imp", "parity"))

    def test_death_is_hidden_until_meeting(self):
        """Un crewmate loin du meurtre ne sait pas que la victime est morte ; il l'apprend à la réunion."""
        w = staged_kill()
        far = 3
        self.assertFalse(w.alive[1])
        self.assertTrue(w.believes_alive(far, 1))
        self.assertFalse(w.believes_alive(0, 1))                 # l'imposteur, lui, le sait
        frac = w.tasks_fraction()
        o = O.local_obs(w, far)
        row = o[O.local_obs(w, far).shape[0] - 4 - O.N_OTHERS * O.OTHER_F:][:O.N_OTHERS * O.OTHER_F]
        self.assertEqual(row.reshape(O.N_OTHERS, O.OTHER_F)[(1 - far - 1) % w.n][6], 1.0)   # « vivant » pour lui
        w.start_meeting(far, emergency=True)
        self.assertFalse(w.believes_alive(far, 1))
        self.assertEqual(w.tasks_fraction(), frac)               # la barre de tâches ne trahit pas le meurtre

    def test_seen_body_is_known(self):
        w = staged_kill(witness=2)
        self.assertFalse(w.believes_alive(2, 1))                 # témoin du meurtre

    def test_claims_and_cross_checks(self):
        """Le témoin a vu l'imposteur en Cafétéria : si l'imposteur dit avoir été ailleurs, il est contredit.
        Les crewmates loin disent la vérité et se confirment entre eux."""
        w = staged_kill(witness=2)
        w.claim_callbacks[0] = lambda world, imp, ctx: (FAR, 3)  # ment : « j'étais en FAR » et accuse 3
        w.vote_callbacks = {p: (lambda env, v, ctx: None) for p in range(w.n)}
        w.start_meeting(2)
        ev = [e for e in w.events if e["type"] == "meeting"][-1]
        ctx = ev["ctx"]
        self.assertEqual(ctx.kill_room, 0)
        self.assertEqual(ctx.claims[0], FAR)
        self.assertEqual(ctx.claims[2], 0)                       # le témoin était en Cafétéria
        self.assertEqual(ctx.claims[3], FAR)
        self.assertIn(2, ctx.contra[0])                          # le témoin l'a vu ailleurs que là où il le dit
        self.assertTrue(set(ctx.absent[0]) >= {3, 4, 5})         # ceux de FAR ne l'y ont pas vu
        self.assertEqual(ctx.accuse, {2: 0, 0: 3})               # témoin -> imposteur ; imposteur -> 3 (mensonge)
        self.assertTrue(set(ctx.confirm[3]) >= {4, 5})
        self.assertIn(3, ctx.cleared)
        self.assertNotIn(0, ctx.cleared)
        # l'observation de vote de 4 reflète tout cela
        o = V.vote_obs(w, 4, ctx)
        row = lambda q: o[V.N_GLOBAL:].reshape(V.N_SLOTS, V.VOTE_F)[(q - 4 - 1) % w.n]
        self.assertEqual(row(0)[15], 1.0)                        # a déclaré
        self.assertGreater(row(0)[19], 0)                        # contredit
        self.assertGreater(row(0)[20], 0)                        # absent
        self.assertEqual(row(0)[22], 1.0)                        # 4 était en FAR et ne l'y a pas vu
        self.assertGreater(row(3)[2], 0)                         # 3 est accusé…
        self.assertEqual(row(0)[24 + (3 - 4 - 1) % w.n], 1.0)    # …par 0

    def test_imperfect_memory(self):
        """Mémoire 0 : personne ne rapporte ce qu'il a vu (ni confirmation ni contradiction), mais le témoin direct
        accuse toujours le tueur."""
        w = staged_kill(witness=2, recall=0.0)
        w.claim_callbacks[0] = lambda world, imp, ctx: (FAR, None)
        w.vote_callbacks = {p: (lambda env, v, ctx: None) for p in range(w.n)}
        w.start_meeting(2)
        ctx = [e for e in w.events if e["type"] == "meeting"][-1]["ctx"]
        self.assertFalse(any(ctx.confirm.values()) or any(ctx.contra.values()) or any(ctx.absent.values()))
        self.assertEqual(ctx.accuse, {2: 0})
        self.assertEqual(ctx.cleared, set())

    def test_meeting_takes_time(self):
        """Une réunion consomme du chrono ; au temps limite, l'imposteur gagne."""
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=3)
        w.reset(imposter=0)
        w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
        w.vote_callbacks = {p: (lambda env, v, ctx: None) for p in range(w.n)}
        w.time = 40.0
        w.start_meeting(2, emergency=True)
        self.assertAlmostEqual(w.time, 40.0 + w.rt.meeting_time)
        self.assertEqual(w.last_meeting_t, w.time)
        w.time = w.rt.max_time - 1.0
        w.emergencies_left[3] = 1
        w.start_meeting(3, emergency=True)
        self.assertEqual((w.winner, w.reason), ("imp", "timeout"))

    def test_endgame_start(self):
        """Entraînement ciblé : la partie démarre en fin de partie, joueurs placés debout, règles intactes."""
        for seed in range(20):
            w = RTWorld(Config(), seed=seed)
            w.reset()
            w.set_controllers(ScriptedCrew(), ScriptedImposter(seed))
            nc, ni = w.start_endgame()
            self.assertEqual((len(w.crew_alive()), len(w.imps_alive())), (nc, ni))
            self.assertIsNone(w._outcome())                       # pas déjà finie
            self.assertTrue(all(M.can_stand(w.x[p], w.y[p]) for p in range(w.n) if w.alive[p]))
            self.assertGreaterEqual(w.time, 60.0)
            r = run_game(RTWorld(Config(), seed=seed), ScriptedCrew(), ScriptedImposter(seed), endgame=True)
            self.assertIn(r["winner"], ("crew", "imp"))

    def test_heuristic_claim_avoids_crime_room(self):
        w = staged_kill()
        w.start_meeting(3, emergency=True)
        ctx = [e for e in w.events if e["type"] == "meeting"][-1]["ctx"]
        self.assertNotEqual(ctx.claims[0], ctx.kill_room)

    def test_emergency_meeting(self):
        w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=3)
        w.reset(imposter=0)
        w.set_controllers(ScriptedCrew(), ScriptedImposter(2))
        self.assertNotIn(EMERGENCY, w.legal_actions(2))          # trop tôt
        w.time = w.rt.emergency_cooldown + 1
        self.assertNotIn(EMERGENCY, w.legal_actions(2))          # pas avant emergency_first (45 s de jeu)
        self.assertGreater(w.emergency_wait(), 0)
        w.time = w.rt.emergency_first + 1
        self.assertIn(EMERGENCY, w.legal_actions(2))             # en Cafétéria, après le délai
        self.assertEqual(w.emergency_wait(), 0)
        w.x[3], w.y[3] = M.task_stand(FAR)
        w._update_rooms()
        self.assertNotIn(EMERGENCY, w.legal_actions(3))          # hors de la Cafétéria
        w.ag[2]["macro"] = ("emergency",)
        w._ai_tick(2, 0.1)
        ev = [e for e in w.events if e["type"] == "meeting"]
        self.assertEqual((len(ev), ev[0]["emergency"], ev[0]["reporter"]), (1, True, 2))
        self.assertEqual(w.emergencies_left[2], 0)
        self.assertEqual(w.stats["emergencies"], 1)
        w.time += w.rt.emergency_cooldown + 1
        self.assertNotIn(EMERGENCY, w.legal_actions(2))          # une seule par joueur

    def test_claim_network(self):
        co, cc = C.dims()
        w = staged_kill(witness=2)
        ctrl = C.ClaimCtrl(NumpyActor(actor(co, C.N_CLAIM)), seed=0)
        w.set_controllers(ScriptedCrew(), ScriptedImposter(2), record_imp=True, imp_claim=ctrl)
        w.vote_callbacks = {p: (lambda env, v, ctx: 0 if v != 0 else None) for p in range(w.n)}
        w.start_meeting(2)
        ctx = [e for e in w.events if e["type"] == "meeting"][-1]["ctx"]
        self.assertIn(0, ctx.claims)
        recs = w.ctraj[0]
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((len(r["obs"]), len(r["crit"])), (co, cc))
        self.assertTrue(r["mask"][r["act"]])
        self.assertEqual(C.decode(w, 0, r["act"])[0], ctx.claims[0])
        self.assertAlmostEqual(r["r"], -w.rw.vote_eject_imp)       # éjecté : la déclaration est punie
        m = C.claim_mask(w, 0).reshape(-1, C.N_ACC)
        self.assertFalse(m[:, (1 - 0 - 1) % w.n].any())            # on n'accuse pas un mort

    def test_claim_learner_update(self):
        try:
            from amongus.ai.mappo import RoleLearner
        except ImportError:
            self.skipTest("PyTorch non installé")
        from amongus.ai.rollout import _stack
        co, cc = C.dims()
        oc, oi, _, _ = O.dims()
        trajs = []
        for seed in range(40):
            w = RTWorld(Config(n_crew=6, n_imp=1, crew_left_to_lose=2), seed=seed)
            run_game(w, ScriptedCrew(), ScriptedImposter(seed), record_imp=True,
                     imp_claim=C.ClaimCtrl(NumpyActor(actor(co, C.N_CLAIM)), seed=seed))
            if w.ctraj[w.imp]:
                trajs.append(_stack(w.ctraj[w.imp]))
            if len(trajs) >= 3:
                break
        self.assertTrue(trajs)
        st = RoleLearner(co, cc, hidden=16, n_actions=C.N_CLAIM).update(trajs, epochs=1, mb=16)
        self.assertGreater(st["n"], 0)

    def test_save_load_claim_and_version_check(self):
        oc, oi, _, _ = O.dims()
        co, _ = C.dims()
        with tempfile.TemporaryDirectory() as d:
            f, g = os.path.join(d, "m.npz"), os.path.join(d, "old.npz")
            save_actors(f, actor(oc, N_ACTIONS), actor(oi, N_ACTIONS), imp_claim=actor(co, C.N_CLAIM))
            self.assertIsNotNone(load_claim_actor(f))
            save_actors(g, actor(oc - 3, N_ACTIONS), actor(oi - 3, N_ACTIONS))
            self.assertIsNone(load_claim_actor(g))
            with self.assertRaises(ValueError):                    # modèle d'une autre version du jeu
                load_actors(g)


if __name__ == "__main__":
    unittest.main()
