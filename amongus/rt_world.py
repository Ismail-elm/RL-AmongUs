"""Monde TEMPS RÉEL : déplacement libre, secondes, collisions, vision.

Architecture
------------
* Niveau haut (IA apprise, MAPPO) : un contrôleur choisit une macro-action quand la précédente est finie :
  « aller dans la salle voisine j », « attendre », « faire ma tâche », « tuer ».
* Niveau bas (navigation) : champ de distance (Dijkstra) + collisions pour marcher dans les couloirs.
* Le monde garde, pour chaque joueur, une MÉMOIRE de ce qu'il a vu (où et quand), et enregistre,
  si on le lui demande, les trajectoires (observation, action, récompense) pour l'apprentissage.
Les règles (tâches, kill, réunion, vote, victoire) viennent du moteur de règles (env.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import rt_map as M
from .config import Config
from .env import AmongUsEnv, Kill, KILL, N_ROOMS, REPORT, TASK, WAIT


@dataclass
class RTConfig:
    step_sec: float = 3.5          # durée d'un « pas » de référence (cooldown, escompte)
    speed: float = 240.0           # px/s
    task_time: float = 8.0         # s de travail sur une borne
    kill_cooldown_s: float = 9.0
    kill_range: float = 120
    report_range: float = 330.0
    use_range: float = 120.0
    witness_range: float = 480.0
    vision_range: float = 470.0    # rayon de vision (mémoire des IA, comme l'affichage)
    max_time: float = 330.0        # temps limite (s) : sinon victoire de l'imposteur
    wait_sec: float = 2.0
    kill_chase_sec: float = 8.0
    sample_dt: float = 1.0         # fréquence du journal de positions (témoignages)
    observe_dt: float = 1.0        # fréquence de mise à jour de la mémoire visuelle des IA
    arrive_margin: float = 60.0


@dataclass
class RewardConfig:
    """Récompenses d'entraînement (n'affectent pas les règles du jeu)."""
    time: float = -0.003           # par seconde, tous les joueurs vivants
    work: float = 0.05             # par seconde de tâche (crewmate qui travaille)
    task_team: float = 1.0         # une tâche finie : récompense partagée par tous les crewmates vivants
    crew_death: float = -1.5       # la victime (tuée ou éjectée à tort)
    crew_death_team: float = -0.3  # les autres crewmates vivants quand un des leurs meurt
    kill: float = 1.5              # imposteur : un kill
    kill_witnessed: float = -0.5   # imposteur : kill vu par au moins un témoin
    wrong_eject: float = 0.5       # imposteur : un crewmate éjecté à tort
    vote_right: float = 1.0        # crewmate : a voté contre l'imposteur (signal individuel, entraînement seulement)
    vote_wrong: float = -0.5       # crewmate : a voté contre un innocent (rentable seulement si P(imposteur) > 1/3)
    vote_eject_imp: float = 5.0    # résultat de la réunion, pour chaque votant : imposteur éjecté (signe inversé pour lui)
    vote_eject_crew: float = -2.0  # résultat de la réunion : un innocent éjecté (signe inversé pour l'imposteur)
    win: float = 10.0
    loss: float = -10.0


class RTWorld(AmongUsEnv):
    def __init__(self, cfg: Config | None = None, rt: RTConfig | None = None, seed: int | None = None,
                 rw: RewardConfig | None = None):
        self.rt = rt or RTConfig()
        self.rw = rw or RewardConfig()
        self.crew_ctrl = None
        self.imp_ctrl = None
        self.crew_vote = None          # contrôleurs de vote appris (None -> règle heuristique de env.py)
        self.imp_vote = None
        self.record_crew = self.record_imp = False
        self.human: int | None = None
        self.human_input = dict(dx=0.0, dy=0.0, use=False)
        super().__init__(cfg, seed)

    # ------------------------------------------------------------------ reset
    def reset(self, imposter: int | None = None):
        super().reset(imposter)
        n, rt = self.n, self.rt
        pts = M.spawn_points(n)
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
        self.face = [1] * n
        self.moving = [False] * n
        self.work: list[dict[int, float]] = [dict() for _ in range(n)]
        self.working = [False] * n
        self.body_pos: dict[int, tuple[float, float]] = {}
        self.ag = {p: dict(macro=None, t0=0.0, last_room=0, pend=None, prog_t=0.0) for p in range(n)}
        self.traj: dict[int, list] = {p: [] for p in range(n)}
        self.vtraj: dict[int, list] = {p: [] for p in range(n)}     # décisions de VOTE enregistrées (une par réunion)
        self._vote_recs: dict[int, dict] = {}                        # votes de la réunion en cours
        self.time = 0.0
        self.cd_s = rt.kill_cooldown_s
        self.cooldown = math.ceil(self.cd_s / rt.step_sec - 1e-9)
        self._sample_t = 0.0
        self._observe_t = 0.0
        self.room = [0] * n
        # mémoire visuelle : dernière position vue de chaque joueur (par observateur)
        NEVER = -1e9
        self.mem_x = [[0.0] * n for _ in range(n)]
        self.mem_y = [[0.0] * n for _ in range(n)]
        self.mem_t = [[NEVER] * n for _ in range(n)]
        self.saw_kill = [set() for _ in range(n)]      # observateur -> tueurs qu'il a vus agir
        self.cleared_known: set[int] = set()           # innocentés par des témoignages (public)
        self._update_rooms()
        return self

    def set_controllers(self, crew, imp, record_crew: bool = False, record_imp: bool = False,
                        crew_vote=None, imp_vote=None):
        """crew / imp : objets avec decide(world, p, legal, record) -> (action, enregistrement|None).
        crew_vote / imp_vote : objets avec vote(world, votant, ctx, record) -> (cible|None, enregistrement|None)."""
        self.crew_ctrl, self.imp_ctrl = crew, imp
        self.crew_vote, self.imp_vote = crew_vote, imp_vote
        self.record_crew, self.record_imp = record_crew, record_imp

    # ------------------------------------------------------- enregistrement (apprentissage)
    def _recording(self, p):
        return self.record_imp if p == self.imp else self.record_crew

    def _credit(self, p, r):
        pend = self.ag[p].get("pend")
        if pend is not None and r:
            pend["r"] += r

    def ai_vote(self, voter, ctx):
        """Vote d'un joueur IA : réseau de vote s'il y en a un pour son rôle, sinon règle heuristique."""
        ctrl = self.imp_vote if voter == self.imp else self.crew_vote
        if ctrl is None:
            return super().ai_vote(voter, ctx)
        rec_flag = self._recording(voter)
        choice, rec = ctrl.vote(self, voter, ctx, rec_flag)
        if rec is not None:
            # signal individuel : sans lui, des votes non coordonnés n'éjectent personne et tout le monde apprend à passer
            rec["r"] = 0.0
            if voter != self.imp and choice is not None:
                rec["r"] = self.rw.vote_right if choice == self.imp else self.rw.vote_wrong
            self._vote_recs[voter] = rec
        return choice

    def _close_votes(self, ejected):
        """Fin de réunion : chaque vote enregistré reçoit le résultat de la réunion et forme un épisode à lui seul.
        (La suite de la partie dépend surtout des déplacements : la créditer au vote noierait le signal.)"""
        rw = self.rw
        for voter, rec in self._vote_recs.items():
            out = 0.0 if ejected is None else (rw.vote_eject_imp if ejected == self.imp else rw.vote_eject_crew)
            rec["r"] += -out if voter == self.imp else out
            rec["dur"], rec["done"] = 1.0, True
            self.vtraj[voter].append(rec)
        self._vote_recs = {}

    def _credit_crew(self, r, exclude=None):
        if not self.record_crew:
            return
        for q in self.crew_alive():
            if q != exclude:
                self._credit(q, r)

    def _close_pending(self, p, done):
        pend = self.ag[p].get("pend")
        if pend is not None:
            pend["dur"] = max(0.25, self.time - pend["t"])
            pend["done"] = done
            self.traj[p].append(pend)
            self.ag[p]["pend"] = None

    def _finish(self, winner, reason, rew, term):
        super()._finish(winner, reason, rew, term)
        rw = self.rw
        for p in range(self.n):
            if not self._recording(p):
                continue
            if self.alive[p] or p == self.imp:
                good = (winner == "imp") == (p == self.imp)
                self._credit(p, rw.win if good else rw.loss)
            self._close_pending(p, True)

    def drain_events(self):
        ev, self.events = self.events, []
        return ev

    # ------------------------------------------------------------ mémoire visuelle
    def sees(self, p, q):
        """p voit-il q (même salle, ou à portée de vision et en ligne de mire) ?"""
        if p == q:
            return True
        rp, rq = self.room[p], self.room[q]
        if rp >= 0 and rp == rq:
            return True
        d = math.hypot(self.x[p] - self.x[q], self.y[p] - self.y[q])
        return d < self.rt.vision_range and M.line_of_sight(self.x[p], self.y[p], self.x[q], self.y[q])

    def _observe(self):
        n, t = self.n, self.time
        for p in range(n):
            if not self.alive[p] or p == self.human:
                continue
            self.mem_x[p][p], self.mem_y[p][p], self.mem_t[p][p] = self.x[p], self.y[p], t
            for q in range(n):
                if q != p and self.alive[q] and self.sees(p, q):
                    self.mem_x[p][q], self.mem_y[p][q], self.mem_t[p][q] = self.x[q], self.y[q], t

    # -------------------------------------------------------------- utilitaires
    def pos(self, p):
        return self.x[p], self.y[p]

    def dist(self, a, b):
        return math.hypot(self.x[a] - self.x[b], self.y[a] - self.y[b])

    def code(self, p):
        return -1 if not self.alive[p] else self.room[p]

    def _update_rooms(self):
        for p in range(self.n):
            if not self.alive[p]:
                continue
            r = M.room_at(self.x[p], self.y[p])
            self.room[p] = r
            if r >= 0:
                self.ag[p]["last_room"] = r
        if self.alive[self.imp]:
            r = self.room[self.imp]
            if r >= 0:
                if any(q != self.imp and self.alive[q] and self.room[q] == r for q in range(self.n)):
                    self.imp_mem = r
                elif self.imp_mem == r:
                    self.imp_mem = None

    def _move(self, p, mx, my):
        x, y = self.x[p], self.y[p]
        nx, ny = x + mx, y + my
        if M.can_stand(nx, ny):
            self.x[p], self.y[p] = nx, ny
        elif M.can_stand(nx, y):
            self.x[p] = nx
        elif M.can_stand(x, ny):
            self.y[p] = ny
        else:
            return
        if abs(mx) > 0.05:
            self.face[p] = 1 if mx > 0 else -1
        self.moving[p] = True

    def _steer(self, p, goal, dt):
        dx, dy = M.steer(M.field(goal), self.x[p], self.y[p])
        d = math.hypot(dx, dy)
        if d > 1e-6:
            s = self.rt.speed * dt / d
            self._move(p, dx * s, dy * s)

    def _toward(self, p, tx, ty, dt):
        dx, dy = tx - self.x[p], ty - self.y[p]
        d = math.hypot(dx, dy)
        if d > 1e-6:
            s = min(self.rt.speed * dt, d) / d
            self._move(p, dx * s, dy * s)

    # ------------------------------------------------------------ règles de fin
    def _outcome(self):
        if not self.alive[self.imp]:
            return "crew", "ejected"
        ac = self.crew_alive()
        if len(ac) <= self.cfg.crew_left_to_lose:
            return "imp", "parity"
        if all(not self.tasks[p] for p in ac):
            return "crew", "tasks"
        if self.time >= self.rt.max_time:
            return "imp", "timeout"
        return None

    def tasks_fraction(self):
        """Barre de tâches globale (comme dans le vrai jeu)."""
        total = self.cfg.tasks_per_crew * len(self.crew_alive())
        left = sum(len(self.tasks[p]) for p in self.crew_alive())
        return 1.0 - left / total if total else 1.0

    # ------------------------------------------------------------------ actions
    def do_kill(self, killer: int, victim: int):
        cfg, rt, rw = self.cfg, self.rt, self.rw
        kr = self.room[killer] if self.room[killer] >= 0 else M.nearest_room(self.x[killer], self.y[killer])
        witnesses = []
        for q in self.crew_alive():
            if q == victim:
                continue
            same = self.room[q] == self.room[killer] and self.room[q] >= 0
            seen = self.dist(q, killer) < rt.witness_range and M.line_of_sight(
                self.x[q], self.y[q], self.x[killer], self.y[killer])
            if same or seen:
                witnesses.append(q)
        for w in witnesses:
            self.saw_kill[w].add(killer)
        self.alive[victim] = False
        self.working[victim] = False
        self.body_pos[victim] = (self.x[victim], self.y[victim])
        self.cd_s = rt.kill_cooldown_s
        self.cooldown = math.ceil(self.cd_s / rt.step_sec - 1e-9)
        self.room[victim] = -1
        # le journal de positions contient le moment du meurtre (victime = -1)
        self.log.append(tuple(self.code(p) for p in range(self.n)))
        self.kills.append(Kill(idx=len(self.log) - 1, room=kr, victim=victim, witnesses=witnesses))
        self.bodies.append((kr, victim))
        self.stats["kills"] += 1
        if witnesses:
            self.stats["witnessed_kills"] += 1
        self.events.append(dict(type="kill", killer=killer, victim=victim, room=kr, witnesses=witnesses,
                                pos=self.body_pos[victim]))
        if self.record_crew or self.record_imp:
            if self.record_imp:
                self._credit(killer, rw.kill + (rw.kill_witnessed if witnesses else 0.0))
            if self.record_crew:
                self._credit(victim, rw.crew_death)
                self._close_pending(victim, True)
                self._credit_crew(rw.crew_death_team)
        out = self._outcome()
        if out is not None:
            self._finish(out[0], out[1], [0.0] * self.n, [False] * self.n)

    def start_meeting(self, reporter: int):
        rw = self.rw
        rew, term = [0.0] * self.n, [False] * self.n
        self._meeting(reporter, rew, term)    # preuves, votes, éjection (env.py)
        ev = self.events[-1]
        self._close_votes(ev["ejected"])
        self.cleared_known |= {c for c in ev["cleared"] if self.alive[c]}
        for p in range(self.n):                # les éjectés : fin de trajectoire
            if term[p] and not self.alive[p] and p != self.imp:
                if self.record_crew:
                    self._credit(p, rw.crew_death)
                    self._close_pending(p, True)
                    self._credit_crew(rw.crew_death_team)
                if self.record_imp:
                    self._credit(self.imp, rw.wrong_eject)
        pts = M.spawn_points(self.n)
        for p in range(self.n):
            self.x[p], self.y[p] = pts[p]
            self.moving[p] = False
            self.working[p] = False
            self.ag[p]["macro"] = None
        self.body_pos.clear()
        self.cd_s = self.rt.kill_cooldown_s
        self.cooldown = math.ceil(self.cd_s / self.rt.step_sec - 1e-9)
        self._sample_t = 0.0
        self._update_rooms()
        out = self._outcome()
        if out is not None:
            self._finish(out[0], out[1], [0.0] * self.n, [False] * self.n)

    def nearest_body(self, p):
        best, bd = None, 1e18
        for (_r, v) in self.bodies:
            bx, by = self.body_pos.get(v, (None, None))
            if bx is None:
                continue
            d = math.hypot(bx - self.x[p], by - self.y[p])
            if d < bd and M.line_of_sight(self.x[p], self.y[p], bx, by):
                best, bd = v, d
        return best, bd

    # ----------------------------------------------------------- actions humaines
    def can_report(self, p):
        """p est-il assez près d'un corps (et le voit-il) pour le signaler ? Vaut aussi pour l'imposteur."""
        if not self.bodies or not self.alive[p]:
            return False
        v, d = self.nearest_body(p)
        return v is not None and d <= self.rt.report_range

    def legal_actions(self, p):
        acts = super().legal_actions(p)
        if p == self.imp and self.can_report(p):
            acts.append(REPORT)        # les crewmates IA signalent automatiquement ; l'imposteur, lui, choisit
        return acts

    def human_can_report(self):
        h = self.human
        return h is not None and self.can_report(h)

    def human_report(self):
        if self.winner is None and self.human_can_report():
            self.start_meeting(self.human)
            return True
        return False

    def kill_target_for(self, killer):
        best, bd = None, 1e18
        for q in self.crew_alive():
            d = self.dist(killer, q)
            if d < bd:
                best, bd = q, d
        return (best, bd) if best is not None else (None, 1e18)

    def human_can_kill(self):
        h = self.human
        if h is None or h != self.imp or not self.alive[h] or self.cd_s > 0:
            return False
        t, d = self.kill_target_for(h)
        return t is not None and d <= self.rt.kill_range

    def human_kill(self):
        if self.winner is None and self.human_can_kill():
            t, _ = self.kill_target_for(self.human)
            self.do_kill(self.human, t)
            return True
        return False

    def my_task_nearby(self, p):
        """Salle de la tâche dont la borne est à portée (None sinon)."""
        r = self.room[p]
        if r < 0 or r not in self.tasks[p]:
            return None
        tx, ty = M.task_terminal(r)
        return r if math.hypot(tx - self.x[p], ty - self.y[p]) <= self.rt.use_range else None

    # ------------------------------------------------------------------- un tick
    def update(self, dt: float):
        if self.winner is not None:
            return
        rt = self.rt
        self.time += dt
        self.t = int(self.time)
        self.cd_s = max(0.0, self.cd_s - dt)
        self.cooldown = math.ceil(self.cd_s / rt.step_sec - 1e-9)
        for p in range(self.n):
            self.moving[p] = False
        if self.record_crew or self.record_imp:
            rs = self.rw.time * dt
            for p in range(self.n):
                if self.alive[p]:
                    self._credit(p, rs)
        for p in range(self.n):
            if not self.alive[p] and p != self.human:
                continue
            if p == self.human:
                self._human_tick(p, dt)
            elif self.alive[p]:
                self._ai_tick(p, dt)
            if self.winner is not None:
                return
        self._update_rooms()
        self._observe_t += dt
        if self._observe_t >= rt.observe_dt:
            self._observe_t = 0.0
            self._observe()
        self._sample_t += dt
        if self._sample_t >= rt.sample_dt:
            self._sample_t = 0.0
            self.log.append(tuple(self.code(p) for p in range(self.n)))
        # signalement automatique par les crewmates IA
        if self.bodies:
            for q in self.crew_alive():
                if q == self.human:
                    continue
                v, d = self.nearest_body(q)
                if v is not None and d <= rt.report_range:
                    self.start_meeting(q)
                    break
        if self.winner is None:
            out = self._outcome()
            if out is not None:
                self._finish(out[0], out[1], [0.0] * self.n, [False] * self.n)

    # --------------------------------------------------------------------- humain
    def _human_tick(self, p, dt):
        inp = self.human_input
        dx, dy = inp["dx"], inp["dy"]
        n = math.hypot(dx, dy)
        sp = self.rt.speed * (1.0 if self.alive[p] else 1.25)
        if n > 0:
            self._move(p, dx / n * sp * dt, dy / n * sp * dt)
        self.working[p] = False
        if self.alive[p] and p != self.imp and inp["use"]:
            r = self.my_task_nearby(p)
            if r is not None:
                self.working[p] = True
                self._work(p, r, dt)

    def _work(self, p, r, dt):
        w = self.work[p]
        w[r] = w.get(r, 0.0) + dt
        self.ag[p]["prog_t"] = self.time
        if self._recording(p):
            self._credit(p, self.rw.work * dt)
        if w[r] >= self.rt.task_time:
            self.tasks[p].discard(r)
            w.pop(r, None)
            self.stats["tasks_done"] += 1
            self.working[p] = False
            self.events.append(dict(type="task", player=p, room=r))
            if self.record_crew:
                self._credit_crew(self.rw.task_team)
            return True
        return False

    # ------------------------------------------------------------------------- IA
    def _decide(self, p):
        rt = self.rt
        legal = self.legal_actions(p)
        ctrl = self.imp_ctrl if p == self.imp else self.crew_ctrl
        rec_flag = self._recording(p)
        a, rec = ctrl.decide(self, p, legal, rec_flag)
        if rec_flag:
            self._close_pending(p, False)
            if rec is not None:
                rec["r"] = 0.0
                rec["t"] = self.time
                self.ag[p]["pend"] = rec
        if a < N_ROOMS:
            return ("goto", a)
        if a == WAIT:
            return ("wait", self.time + rt.wait_sec)
        if a == TASK:
            return ("task", self.room[p], 0)
        if a == REPORT:
            return ("report",)
        cands = [q for q in self.crew_alive() if self.room[q] == self.room[p]]
        if not cands:
            return ("wait", self.time + 0.5)
        tgt = min(cands, key=lambda q: self.dist(p, q))
        return ("kill", tgt, self.time + rt.kill_chase_sec)

    def _ai_tick(self, p, dt):
        rt, ag = self.rt, self.ag[p]
        m = ag["macro"]
        if m is None:
            if self.room[p] < 0:
                m = ("goto", ag["last_room"])
            else:
                m = self._decide(p)
            ag["macro"], ag["t0"] = m, self.time
        kind = m[0]
        done = False
        self.working[p] = False
        if kind == "goto":
            j = m[1]
            if M.inside_room(self.x[p], self.y[p], j, rt.arrive_margin) or self.time - ag["t0"] > 16:
                done = True
            else:
                self._steer(p, ("room", j), dt)
        elif kind == "wait":
            done = self.time >= m[1]
        elif kind == "task":
            r = m[1]
            sx, sy = M.task_stand(r)
            if math.hypot(sx - self.x[p], sy - self.y[p]) > 40 and m[2] == 0:
                self._steer(p, ("spot", r), dt)
                if self.time - ag["t0"] > 16:
                    done = True
            else:
                ag["macro"] = m = ("task", r, 1)
                self.working[p] = True
                if self._work(p, r, dt):
                    done = True
        elif kind == "report":
            done = True
            if self.can_report(p):
                self.start_meeting(p)
        elif kind == "kill":
            tgt = m[1]
            if (not self.alive[tgt]) or self.room[tgt] != self.room[p] or self.time > m[2]:
                done = True
            else:
                self._toward(p, self.x[tgt], self.y[tgt], dt)
                if self.dist(p, tgt) <= rt.kill_range and self.cd_s <= 0:
                    self.do_kill(p, tgt)
                    done = True
        if done:
            ag["macro"] = None


def run_game(world: RTWorld, crew, imp, dt: float = 0.1, imposter: int | None = None,
             max_ticks: int = 20000, record_crew: bool = False, record_imp: bool = False,
             crew_vote=None, imp_vote=None):
    """Joue une partie complète sans affichage. Renvoie des statistiques (trajectoires dans world.traj / vtraj)."""
    world.reset(imposter)
    world.set_controllers(crew, imp, record_crew, record_imp, crew_vote, imp_vote)
    n = 0
    while world.winner is None and n < max_ticks:
        world.update(dt)
        world.drain_events()
        n += 1
    st = world.stats
    return dict(winner=world.winner or "imp", reason=world.reason or "timeout", seconds=world.time,
                kills=st["kills"], meetings=st["meetings"], good_ejections=st["good_ejections"],
                wrong_ejections=st["wrong_ejections"], tasks_done=st["tasks_done"],
                witnessed_kills=st["witnessed_kills"], skipped_votes=st["skipped_votes"])
