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
from .env import EMERGENCY, FIX, SABOTAGE, VENT_A, VENT_B, AmongUsEnv, Kill, KILL, N_ROOMS, REPORT, TASK, WAIT
from .mapdata import ADJ


@dataclass
class RTConfig:
    step_sec: float = 3.5          # durée d'un « pas » de référence (cooldown, escompte)
    speed: float = 240.0           # px/s
    task_time: float = 5.0         # s de travail sur une borne (8 avant)
    kill_cooldown_s: float = 7.0   # 9 s avant : sans l'abus du bouton d'urgence, les crewmates gagnaient ~73 %
    kill_range: float = 120
    report_range: float = 330.0
    use_range: float = 120.0
    witness_range: float = 480.0
    vision_range: float = 470.0    # rayon de vision de l'imposteur (mémoire des IA, comme l'affichage)
    crew_vision_range: float = 420.0   # rayon de vision des crewmates (vrai jeu : plus petit que celui de l'imposteur) ; 470 avant
    together_range: float = 300.0  # deux joueurs vus à moins de cette distance l'un de l'autre sont « ensemble »
    emergencies: int = 1           # réunions d'urgence par joueur et par partie (bouton de la Cafétéria)
    emergency_total: int = 99      # réunions d'urgence au total dans la partie (tous joueurs confondus)
    emergency_cooldown: float = 15.0   # s depuis la dernière réunion avant de pouvoir l'appeler
    emergency_first: float = 45.0      # s de jeu avant la première urgence possible (avant : personne n'a rien vu, le
                                       # vote se fait au hasard ; les imposteurs s'en servaient pour faire éjecter un innocent)
    meeting_time: float = 20.0     # s de chrono consommées par chaque réunion (discussion et vote, comme dans le vrai jeu)
    lights_vision: float = 160.0   # vision des crewmates quand les lumières sont coupées
    sabotage_cooldown: float = 30.0    # s entre deux sabotages (et au début de la partie / après une réunion)
    lights_max: float = 45.0       # les lumières se rallument seules au bout de ce temps
    fix_time: float = 3.0          # s de réparation au panneau d'Électricité
    vent_time: float = 1.0         # s passées dans le conduit (invisible)
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
    task_team: float = 2.0         # une tâche finie : récompense partagée par tous les crewmates vivants (1 avant)
    crew_death: float = -1.5       # la victime (tuée ou éjectée à tort)
    crew_death_team: float = -0.3  # les autres crewmates vivants quand un des leurs meurt
    kill: float = 1.5              # imposteur : un kill
    kill_witnessed: float = -0.5   # imposteur : kill vu par au moins un témoin
    wrong_eject: float = 0.5       # imposteur : un crewmate éjecté à tort
    vote_right: float = 1.0        # crewmate : a voté contre l'imposteur (signal individuel, entraînement seulement)
    vote_wrong: float = -1.0       # crewmate : a voté contre un innocent. Voter n'est rentable que si P(imposteur) > 1/2 :
                                   # sans indice (P = 1/3 avec 2 imposteurs sur 6), mieux vaut passer
    vote_eject_imp: float = 5.0    # résultat de la réunion, pour chaque votant : imposteur éjecté (signe inversé pour lui)
    vote_eject_crew: float = -2.0  # résultat de la réunion : un innocent éjecté (signe inversé pour l'imposteur)
    imp_ejected: float = -5.0      # imposteur éjecté (sa trajectoire s'arrête ; son partenaire peut encore gagner)
    fix: float = 0.5               # crewmate : a réparé les lumières
    vent: float = -0.05            # imposteur : chaque passage par un conduit (sans coût, l'IA en abusait comme « attente » :
                                   # ~28 allers-retours en 50 s en fin de partie au lieu d'aller tuer)
    win: float = 10.0
    loss: float = -10.0


class RTWorld(AmongUsEnv):
    def __init__(self, cfg: Config | None = None, rt: RTConfig | None = None, seed: int | None = None,
                 rw: RewardConfig | None = None):
        self.rt = rt or RTConfig()
        self.rw = rw or RewardConfig()
        self.crew_ctrl = None
        self.imp_ctrl = None
        self.crew_vote = None          # contrôleurs de vote appris (None -> règle heuristique)
        self.imp_vote = None
        self.imp_claim = None          # déclaration appris de l'imposteur (None -> règle heuristique)
        self.claim_callbacks: dict[int, callable] = {}   # imposteur humain : f(monde, joueur, ctx) -> (salle, accusé)
        self.record_crew = self.record_imp = False
        self.human: int | None = None
        self.human_input = dict(dx=0.0, dy=0.0, use=False)
        super().__init__(cfg, seed)

    def _n_imp(self):
        return self.cfg.n_imp

    # ------------------------------------------------------------------ reset
    def reset(self, imposter=None):
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
        self.ctraj: dict[int, list] = {p: [] for p in range(n)}     # déclarations enregistrées des imposteurs
        self._claim_recs: dict[int, dict] = {}
        self.known_dead: set[int] = set()              # morts connues de tous (révélées par une réunion)
        self.seen_dead = [set() for _ in range(n)]     # morts qu'un joueur a vues lui-même (corps, meurtre)
        self.emergencies_left = [rt.emergencies] * n
        self.emergency_total_left = rt.emergency_total
        self.last_meeting_t = 0.0
        self._emergency_now = False
        self.lights_off = False
        self.lights_t = 0.0                            # début du sabotage en cours
        self.sab_ready_t = rt.sabotage_cooldown        # instant où l'imposteur pourra saboter
        self.fix_prog = 0.0
        self.vent_seen: dict[int, list] = {}           # imposteur -> crewmates qui l'ont vu entrer/sortir d'un conduit
        self.in_vent: set[int] = set()                 # imposteurs en train de traverser un conduit (invisibles)
        self.vent_idx = 0                              # humain : prochaine sortie du conduit
        self.time = 0.0
        self.kcd = [rt.kill_cooldown_s] * n            # recharge du meurtre, par imposteur
        self._sample_t = 0.0
        self._observe_t = 0.0
        self.room = [0] * n
        # mémoire visuelle : dernière position vue de chaque joueur (par observateur)
        NEVER = -1e9
        self.mem_x = [[0.0] * n for _ in range(n)]
        self.mem_y = [[0.0] * n for _ in range(n)]
        self.mem_t = [[NEVER] * n for _ in range(n)]
        self.saw_kill = [set() for _ in range(n)]      # observateur -> tueurs qu'il a vus agir
        # pair_t[p][q][v] : dernière fois que p a vu q et v ENSEMBLE (« j'ai vu Rouge avec la victime »)
        self.pair_t = [[[NEVER] * n for _ in range(n)] for _ in range(n)]
        self.cleared_known: set[int] = set()           # innocentés par des témoignages (public)
        self._update_rooms()
        return self

    def set_controllers(self, crew, imp, record_crew: bool = False, record_imp: bool = False,
                        crew_vote=None, imp_vote=None, imp_claim=None):
        """crew / imp : objets avec decide(world, p, legal, record) -> (action, enregistrement|None).
        crew_vote / imp_vote : objets avec vote(world, votant, ctx, record) -> (cible|None, enregistrement|None).
        imp_claim : objet avec claim(world, imposteur, ctx, record) -> (salle, accusé|None, enregistrement|None)."""
        self.crew_ctrl, self.imp_ctrl = crew, imp
        self.crew_vote, self.imp_vote = crew_vote, imp_vote
        self.imp_claim = imp_claim
        self.record_crew, self.record_imp = record_crew, record_imp

    # ------------------------------------------------------- enregistrement (apprentissage)
    def _recording(self, p):
        return self.record_imp if p in self.imps else self.record_crew

    @property
    def cd_s(self):
        """Recharge du meurtre de l'imposteur suivi (humain s'il est imposteur, sinon le premier)."""
        p = self.human if (self.human is not None and self.human in self.imps) else self.imp
        return self.kcd[p]

    def _credit(self, p, r):
        pend = self.ag[p].get("pend")
        if pend is not None and r:
            pend["r"] += r

    def ai_vote(self, voter, ctx):
        """Vote d'un joueur IA : réseau de vote s'il y en a un pour son rôle, sinon règle heuristique."""
        ctrl = self.imp_vote if voter in self.imps else self.crew_vote
        if ctrl is None:
            return self.heuristic_vote(voter, ctx)
        rec_flag = self._recording(voter)
        choice, rec = ctrl.vote(self, voter, ctx, rec_flag)
        if rec is not None:
            # signal individuel : sans lui, des votes non coordonnés n'éjectent personne et tout le monde apprend à passer
            rec["r"] = 0.0
            if voter not in self.imps and choice is not None:
                rec["r"] = self.rw.vote_right if choice in self.imps else self.rw.vote_wrong
            self._vote_recs[voter] = rec
        return choice

    def _close_votes(self, ejected):
        """Fin de réunion : chaque vote enregistré reçoit le résultat de la réunion et forme un épisode à lui seul.
        (La suite de la partie dépend surtout des déplacements : la créditer au vote noierait le signal.)"""
        rw = self.rw
        out = 0.0 if ejected is None else (rw.vote_eject_imp if ejected in self.imps else rw.vote_eject_crew)
        for voter, rec in self._vote_recs.items():
            rec["r"] += -out if voter in self.imps else out
            rec["dur"], rec["done"] = 1.0, True
            self.vtraj[voter].append(rec)
        self._vote_recs = {}
        for imp, rec in self._claim_recs.items():      # déclarations des imposteurs : même résultat que leur vote
            rec["r"] -= out
            rec["dur"], rec["done"] = 1.0, True
            self.ctraj[imp].append(rec)
        self._claim_recs = {}

    # ------------------------------------------------------------- réunion : discussion
    def zone(self, p):
        """Salle où se trouve p (dans un couloir : la salle la plus proche)."""
        r = self.room[p]
        return r if r >= 0 else M.nearest_room(self.x[p], self.y[p])

    def _evidence(self):
        """Témoins directs encore en vie. (Les innocentés viennent des déclarations : voir _prepare_meeting.)"""
        wit = []
        for w in self.witness_of():
            if w not in wit:
                wit.append(w)
        return set(), wit

    def witness_of(self):
        """Témoins directs vivants -> imposteur qu'ils ont vu tuer (ou utiliser un conduit)."""
        out = {}
        for k in self.kills:
            for w in k.witnesses:
                if self.alive[w] and w not in self.imps:
                    out[w] = k.killer
        for imp, ws in self.vent_seen.items():
            for w in ws:
                if self.alive[w] and w not in out:
                    out[w] = imp
        return out

    def _prepare_meeting(self, ctx):
        """Discussion avant le vote, à propos du dernier meurtre depuis la réunion précédente.
        1) Chaque crewmate dit honnêtement où il était ; un témoin direct accuse le tueur.
        2) Chaque imposteur, après les avoir entendus, dit où il était (vrai ou faux) et peut accuser quelqu'un.
        3) Recoupement par les crewmates : « je l'ai vu là » (confirme), « je l'ai vu ailleurs » (contredit),
           « j'étais dans cette salle et il n'y était pas » (absent). Innocenté = confirmé, hors de la salle du crime.
           Un imposteur qui dit avoir été dans la même salle que son partenaire le « confirme » (il ment)."""
        ctx.emergency = self._emergency_now
        self.known_dead |= {p for p in range(self.n) if not self.alive[p]}     # la réunion révèle les morts
        # mémoire imparfaite : chaque crewmate ne se souvient (et ne parle) que d'une partie des joueurs qu'il a vus
        rc = self.cfg.recall
        ctx.shared = {c: {q for q in range(self.n) if q != c and self.rng.random() < rc} for c in self.crew_alive()}
        ref = self.kills[-1] if self.kills else None
        if ref is None:
            ctx.cleared = set()
            return
        ctx.kill_room, ctx.ref = ref.room, ref
        crew = [c for c in self.crew_alive()]
        wit = self.witness_of()
        for c in crew:
            ctx.claims[c] = ref.zone.get(c, self.zone(c))
            if c in wit:
                ctx.accuse[c] = wit[c]
        for imp in self.imps_alive():
            room, acc = self.ai_claim(ctx, imp)
            ctx.claims[imp] = int(room)
            if acc is not None and acc not in self.imps and self.alive[acc]:
                ctx.accuse[imp] = acc
        cleared = set()
        for q, cr in ctx.claims.items():
            conf, contra, absent = [], [], []
            for c in crew:
                if c == q or q not in ctx.shared[c]:
                    continue
                if q in ref.sight.get(c, ()):
                    (conf if ref.zone.get(q) == cr else contra).append(c)
                elif ref.rooms.get(c, -1) == cr:
                    absent.append(c)
            if q in self.imps:                     # un imposteur se porte garant de son partenaire s'il dit avoir été
                for i in self.imps_alive():        # dans la même salle (« on était ensemble ») : mensonge invérifiable
                    if i != q and ctx.claims.get(i) == cr:
                        conf.append(i)
            ctx.confirm[q], ctx.contra[q], ctx.absent[q] = conf, contra, absent
            if conf and not contra and not absent and cr != ref.room:
                cleared.add(q)
        ctx.cleared = cleared

    def ai_claim(self, ctx, imp=None):
        """Déclaration d'un imposteur -> (salle, accusé | None)."""
        imp = self.imp if imp is None else imp
        cb = self.claim_callbacks.get(imp)
        if cb is not None:
            return cb(self, imp, ctx)
        if self.imp_claim is None:
            return self.heuristic_claim(ctx, imp)
        room, acc, rec = self.imp_claim.claim(self, imp, ctx, self._recording(imp))
        if rec is not None:
            rec["r"] = 0.0
            self._claim_recs[imp] = rec
        return room, acc

    def heuristic_claim(self, ctx, imp=None):
        """Règle simple : se dire dans une salle où aucun crewmate ne dit avoir été, loin du crime ; n'accuse personne."""
        taken = set(ctx.claims.values())
        rooms = [r for r in range(N_ROOMS) if r != ctx.kill_room and r not in taken] or \
            [r for r in range(N_ROOMS) if r != ctx.kill_room]
        return self.rng.choice(rooms), None

    def heuristic_vote(self, voter, ctx):
        """Vote par règles (bots, et sans réseau de vote)."""
        alive = self.alive_players()
        others = [p for p in alive if p != voter]
        if voter in self.imps:
            cands = [p for p in others if p not in ctx.cleared and p not in self.imps]
            return self.rng.choice(cands) if cands else None
        wit = self.witness_of()
        if voter in wit:
            return wit[voter]
        liars = [q for q in others if ctx.contra.get(q) or ctx.absent.get(q)]
        if liars:
            return self.rng.choice(liars)
        counts = {}
        for a, t in ctx.accuse.items():
            if a != voter and t in others:
                counts[t] = counts.get(t, 0) + 1
        if counts:
            top = max(counts, key=counts.get)
            if top not in ctx.cleared:
                return top
        cleared = set(ctx.cleared) | self.cleared_known
        suspects = [p for p in others if p not in cleared]
        if not suspects or len(suspects) > self.cfg.skip_threshold:
            return None
        return self.rng.choice(suspects)

    def believes_alive(self, p, q):
        """Ce que p sait : l'imposteur sait qui il a tué ; un crewmate apprend une mort à la réunion ou en la voyant."""
        if p in self.imps or p == q:
            return self.alive[q]
        return q not in self.known_dead and q not in self.seen_dead[p]

    def can_emergency(self, p):
        return (self.alive[p] and self.room[p] == 0 and self.emergencies_left[p] > 0 and self.emergency_total_left > 0
                and self.time >= self.rt.emergency_first
                and self.time - self.last_meeting_t >= self.rt.emergency_cooldown)

    def emergency_wait(self):
        """Secondes avant que le bouton d'urgence soit disponible (0 = maintenant)."""
        return max(0.0, self.rt.emergency_first - self.time,
                   self.rt.emergency_cooldown - (self.time - self.last_meeting_t))

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
            if self.alive[p] or p in self.imps:
                good = (winner == "imp") == (p in self.imps)
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
        if q in self.in_vent:
            return False
        dark = self.lights_off and p not in self.imps
        rp, rq = self.room[p], self.room[q]
        if rp >= 0 and rp == rq and not dark:
            return True
        d = math.hypot(self.x[p] - self.x[q], self.y[p] - self.y[q])
        return d < self.vision(p) and M.line_of_sight(self.x[p], self.y[p], self.x[q], self.y[q])

    def vision(self, p):
        if p in self.imps:
            return self.rt.vision_range
        return self.rt.lights_vision if self.lights_off else self.rt.crew_vision_range

    def _observe(self):
        n, t = self.n, self.time
        for p in range(n):
            if not self.alive[p] or p == self.human:
                continue
            self.mem_x[p][p], self.mem_y[p][p], self.mem_t[p][p] = self.x[p], self.y[p], t
            seen = [p]
            for q in range(n):
                if q != p and self.alive[q] and self.sees(p, q):
                    self.mem_x[p][q], self.mem_y[p][q], self.mem_t[p][q] = self.x[q], self.y[q], t
                    seen.append(q)
            tr, pt = self.rt.together_range, self.pair_t[p]
            for i, a in enumerate(seen):
                for b in seen[i + 1:]:
                    if math.hypot(self.x[a] - self.x[b], self.y[a] - self.y[b]) < tr:
                        pt[a][b] = pt[b][a] = t
            if p not in self.imps:             # corps aperçus : ce crewmate sait que la victime est morte
                for (_r, v) in self.bodies:
                    bp = self.body_pos.get(v)
                    if bp is not None and math.hypot(bp[0] - self.x[p], bp[1] - self.y[p]) < self.vision(p) \
                            and M.line_of_sight(self.x[p], self.y[p], bp[0], bp[1]):
                        self.seen_dead[p].add(v)

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
        for imp in self.imps_alive():                 # mémoire partagée des imposteurs (bots scriptés)
            r = self.room[imp]
            if r >= 0:
                if any(q not in self.imps and self.alive[q] and self.room[q] == r for q in range(self.n)):
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
        dx, dy = M.steer_goal(goal, self.x[p], self.y[p])
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
        if not self.imps_alive():
            return "crew", "ejected"
        ac = self.crew_alive()
        if self._parity(ac):
            return "imp", "parity"
        if all(not self.tasks[p] for p in ac):
            return "crew", "tasks"
        if self.time >= self.rt.max_time:
            return "imp", "timeout"
        return None

    ENDGAMES = ((2, 1, 0.35), (3, 2, 0.25), (3, 1, 0.2), (4, 2, 0.2))   # (crewmates, imposteurs vivants, poids)

    def start_endgame(self, rng=None):
        """Fait démarrer la partie directement en FIN DE PARTIE (entraînement ciblé) : il ne reste que quelques
        crewmates et un ou deux imposteurs, éparpillés dans le vaisseau, au milieu du chrono. Ces situations sont rares
        dans une partie normale et décisives (un meurtre de plus fait gagner l'imposteur) : sans cela, l'IA les apprend mal."""
        rng = rng or self.rng
        r, acc = rng.random() * sum(w for _, _, w in self.ENDGAMES), 0.0
        for n_crew, n_imp, wgt in self.ENDGAMES:
            acc += wgt
            if r <= acc:
                break
        n_imp = min(n_imp, len(self.imps))
        crew = [p for p in range(self.n) if p not in self.imps]
        rng.shuffle(crew)
        imps = list(self.imps)
        rng.shuffle(imps)
        dead = crew[n_crew:] + imps[n_imp:]
        for p in dead:
            self.alive[p] = False
        self.known_dead |= set(dead)
        t0 = rng.uniform(60.0, 210.0)
        self.time, self.last_meeting_t = t0, t0
        rooms = list(range(1, 12))
        for p in range(self.n):
            if self.alive[p]:
                self.x[p], self.y[p] = M.room_center(rng.choice(rooms))
                self.ag[p]["last_room"] = 0
        done = 0
        for p in crew[:n_crew]:                        # une partie des tâches est déjà faite
            ts = sorted(self.tasks[p])
            k = rng.randint(0, max(0, len(ts) - 1))
            for t in rng.sample(ts, k):
                self.tasks[p].discard(t)
            done += k
        self.stats["tasks_done"] = done + self.cfg.tasks_per_crew * (len(crew) - n_crew) // 2
        for i in self.imps:
            self.kcd[i] = rng.uniform(0.0, self.rt.kill_cooldown_s)
        self.sab_ready_t = t0 + rng.uniform(0.0, self.rt.sabotage_cooldown)
        self.emergencies_left = [rng.randint(0, self.rt.emergencies) for _ in range(self.n)]
        self._update_rooms()
        return n_crew, n_imp

    def tasks_fraction(self):
        """Barre de tâches globale : tâches faites / tâches de départ (elle ne bouge pas quand quelqu'un meurt, sinon
        elle trahirait les meurtres)."""
        total = self.cfg.tasks_per_crew * self.cfg.n_crew
        return min(1.0, self.stats["tasks_done"] / total) if total else 1.0

    # ------------------------------------------------------------------ actions
    def do_kill(self, killer: int, victim: int):
        cfg, rt, rw = self.cfg, self.rt, self.rw
        kr = self.room[killer] if self.room[killer] >= 0 else M.nearest_room(self.x[killer], self.y[killer])
        witnesses = [q for q in self.crew_alive() if q != victim and self.sees(q, killer)]
        for w in witnesses:
            self.saw_kill[w].add(killer)
            self.seen_dead[w].add(victim)
        # instantané pour les déclarations en réunion : où était chacun, et qui voyait qui
        snap = dict(t=self.time, zone={}, rooms={}, sight={}, dark=self.lights_off, killer=killer)
        for p in self.alive_players():
            if p == victim:
                continue
            snap["zone"][p] = kr if p == killer else self.zone(p)
            snap["rooms"][p] = self.room[p]
            snap["sight"][p] = {q for q in self.alive_players() if q != p and q != victim and self.sees(p, q)}
        self.alive[victim] = False
        self.working[victim] = False
        self.body_pos[victim] = (self.x[victim], self.y[victim])
        self.kcd[killer] = rt.kill_cooldown_s
        self.room[victim] = -1
        # le journal de positions contient le moment du meurtre (victime = -1)
        self.log.append(tuple(self.code(p) for p in range(self.n)))
        self.kills.append(Kill(idx=len(self.log) - 1, room=kr, victim=victim, witnesses=witnesses, **snap))
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

    def start_meeting(self, reporter: int, emergency: bool = False):
        rw = self.rw
        rew, term = [0.0] * self.n, [False] * self.n
        self._emergency_now = emergency
        if emergency:
            self.emergencies_left[reporter] -= 1
            self.emergency_total_left -= 1
            self.stats["emergencies"] = self.stats.get("emergencies", 0) + 1
        n_ev = len(self.events)
        self._meeting(reporter, rew, term)    # discussion, votes, éjection (env.py + _prepare_meeting)
        ev = [e for e in self.events[n_ev:] if e["type"] == "meeting"][-1]
        self._emergency_now = False
        self.time += self.rt.meeting_time      # la réunion prend du temps : enchaîner les réunions rapproche du temps limite
        self.t = int(self.time)
        self.last_meeting_t = self.time
        self.known_dead |= {p for p in range(self.n) if not self.alive[p]}
        self._close_votes(ev["ejected"])
        self.cleared_known |= {c for c in ev["cleared"] if self.alive[c]}
        for p in range(self.n):                # les éjectés : fin de trajectoire
            if not (term[p] and not self.alive[p]):
                continue
            if p in self.imps:
                if self.record_imp:
                    self._credit(p, rw.imp_ejected)
                    self._close_pending(p, True)
                continue
            if self.record_crew:
                self._credit(p, rw.crew_death)
                self._close_pending(p, True)
                self._credit_crew(rw.crew_death_team)
            if self.record_imp:
                for imp in self.imps_alive():
                    self._credit(imp, rw.wrong_eject)
        pts = M.spawn_points(self.n)
        for p in range(self.n):
            self.x[p], self.y[p] = pts[p]
            self.moving[p] = False
            self.working[p] = False
            self.ag[p]["macro"] = None
        self.body_pos.clear()
        self.vent_seen = {}
        self.in_vent = set()
        self.lights_off = False
        self.fix_prog = 0.0
        self.sab_ready_t = self.time + self.rt.sabotage_cooldown
        self.kcd = [self.rt.kill_cooldown_s] * self.n
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
        r = self.room[p]
        acts = list(ADJ[r]) + [WAIT] if r >= 0 else [WAIT]
        imp = p in self.imps
        if imp:
            if self.kcd[p] <= 0 and any(self.room[q] == r for q in self.crew_alive()):
                acts.append(KILL)      # on ne tue que des crewmates (pas son partenaire)
            if self.can_report(p):
                acts.append(REPORT)    # les crewmates IA signalent automatiquement ; les imposteurs, eux, choisissent
        elif r in self.tasks[p]:
            acts.append(TASK)
        if self.can_emergency(p):
            acts.append(EMERGENCY)
        if imp:
            links = M.VENT_LINKS.get(self.room[p])
            if links:
                acts += [VENT_A, VENT_B]
            if self.can_sabotage():
                acts.append(SABOTAGE)
        elif self.lights_off and self.room[p] == M.FIX_ROOM:
            acts.append(FIX)
        return acts

    # ------------------------------------------------------------- conduits et sabotage
    def can_sabotage(self):
        return bool(self.imps_alive()) and not self.lights_off and self.time >= self.sab_ready_t

    def sabotage(self):
        if not self.can_sabotage():
            return False
        self.lights_off, self.lights_t, self.fix_prog = True, self.time, 0.0
        self.events.append(dict(type="lights", on=False))
        return True

    def _restore_lights(self, fixer=None):
        self.lights_off = False
        self.fix_prog = 0.0
        self.sab_ready_t = self.time + self.rt.sabotage_cooldown
        self.events.append(dict(type="lights", on=True, fixer=fixer))
        if fixer is not None and self.record_crew:
            self._credit(fixer, self.rw.fix)

    def _fix(self, p, dt):
        """p travaille au panneau d'Électricité. -> True quand les lumières sont réparées."""
        self.fix_prog += dt
        if self.fix_prog >= self.rt.fix_time:
            self._restore_lights(p)
            return True
        return False

    def near_fix(self, p):
        fx, fy = M.fix_pos()
        return self.lights_off and self.room[p] == M.FIX_ROOM and math.hypot(fx - self.x[p], fy - self.y[p]) <= self.rt.use_range

    def _vent_witnesses(self, imp):
        """Crewmates qui voient cet imposteur à cet instant (entrée ou sortie d'un conduit)."""
        seen = [q for q in self.crew_alive() if self.sees(q, imp)]
        lst = self.vent_seen.setdefault(imp, [])
        for q in seen:
            self.saw_kill[q].add(imp)
            if q not in lst:
                lst.append(q)
        return seen

    def vent(self, p, target):
        """L'imposteur entre dans la grille de sa salle et ressort par celle de `target`."""
        seen_in = self._vent_witnesses(p)
        src = self.room[p]
        self.x[p], self.y[p] = M.vent_pos(target)
        self.in_vent.add(p)
        self._update_rooms()
        self.events.append(dict(type="vent", player=p, src=src, dst=target, seen=list(seen_in)))
        if self.record_imp and p in self.imps:
            self._credit(p, self.rw.vent)

    def _vent_exit(self, p):
        self.in_vent.discard(p)
        seen = self._vent_witnesses(p)
        if seen:
            self.events.append(dict(type="vent_seen", player=p, seen=list(seen)))

    def human_can_vent(self):
        h = self.human
        if h is None or h not in self.imps or not self.alive[h] or h in self.in_vent:
            return False
        r = self.room[h]
        if r not in M.VENT_LINKS:
            return False
        vx, vy = M.vent_pos(r)
        return math.hypot(vx - self.x[h], vy - self.y[h]) <= self.rt.use_range

    def human_vent(self):
        if self.winner is None and self.human_can_vent():
            links = M.VENT_LINKS[self.room[self.human]]
            self.vent(self.human, links[self.vent_idx % len(links)])
            self.vent_idx += 1
            self._vent_exit(self.human)
            return True
        return False

    def human_sabotage(self):
        h = self.human
        return h is not None and h in self.imps and self.winner is None and self.sabotage()

    def human_can_report(self):
        h = self.human
        return h is not None and self.can_report(h)

    def human_report(self):
        if self.winner is None and self.human_can_report():
            self.start_meeting(self.human)
            return True
        return False

    def human_can_emergency(self):
        return self.human is not None and self.can_emergency(self.human)

    def human_emergency(self):
        if self.winner is None and self.human_can_emergency():
            self.start_meeting(self.human, emergency=True)
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
        if h is None or h not in self.imps or not self.alive[h] or self.kcd[h] > 0:
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
        for imp in self.imps:
            self.kcd[imp] = max(0.0, self.kcd[imp] - dt)
        if self.lights_off and self.time - self.lights_t >= rt.lights_max:
            self._restore_lights()
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
        if self.alive[p] and p not in self.imps and inp["use"]:
            if self.near_fix(p):
                self.working[p] = True
                self._fix(p, dt)
                return
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
        ctrl = self.imp_ctrl if p in self.imps else self.crew_ctrl
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
        if a == EMERGENCY:
            return ("emergency",)
        if a in (VENT_A, VENT_B):
            links = M.VENT_LINKS[self.room[p]]
            return ("vent", self.room[p], links[0 if a == VENT_A else 1])
        if a == SABOTAGE:
            return ("sabotage",)
        if a == FIX:
            return ("fix",)
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
        elif kind == "emergency":
            done = True
            if self.can_emergency(p):
                self.start_meeting(p, emergency=True)
        elif kind == "sabotage":
            done = True
            self.sabotage()
        elif kind == "fix":
            if not self.lights_off:
                done = True
            else:
                fx, fy = M.fix_pos()
                if math.hypot(fx - self.x[p], fy - self.y[p]) > 40:
                    self._steer(p, ("fix", M.FIX_ROOM), dt)
                    if self.time - ag["t0"] > 16:
                        done = True
                else:
                    self.working[p] = True
                    done = self._fix(p, dt)
        elif kind == "vent":
            src, dst = m[1], m[2]
            if len(m) == 3:                            # 1) aller jusqu'à la grille
                vx, vy = M.vent_pos(src)
                if math.hypot(vx - self.x[p], vy - self.y[p]) > 40:
                    self._steer(p, ("vent", src), dt)
                    if self.time - ag["t0"] > 16 or self.room[p] != src:
                        done = True
                else:                                  # 2) traverser le conduit (invisible)
                    self.vent(p, dst)
                    ag["macro"] = ("vent", src, dst, self.time + rt.vent_time)
            elif self.time >= m[3]:                    # 3) ressortir
                self._vent_exit(p)
                done = True
        elif kind == "kill":
            tgt = m[1]
            if (not self.alive[tgt]) or self.room[tgt] != self.room[p] or self.time > m[2]:
                done = True
            else:
                self._toward(p, self.x[tgt], self.y[tgt], dt)
                if self.dist(p, tgt) <= rt.kill_range and self.kcd[p] <= 0:
                    self.do_kill(p, tgt)
                    done = True
        if done:
            ag["macro"] = None


def run_game(world: RTWorld, crew, imp, dt: float = 0.1, imposter=None,
             max_ticks: int = 20000, record_crew: bool = False, record_imp: bool = False,
             crew_vote=None, imp_vote=None, imp_claim=None, endgame: bool = False):
    """Joue une partie complète sans affichage. Renvoie des statistiques (trajectoires dans world.traj / vtraj / ctraj).
    endgame : la partie démarre directement en fin de partie (voir RTWorld.start_endgame)."""
    world.reset(imposter)
    world.set_controllers(crew, imp, record_crew, record_imp, crew_vote, imp_vote, imp_claim)
    if endgame:
        world.start_endgame()
    n = 0
    while world.winner is None and n < max_ticks:
        world.update(dt)
        world.drain_events()
        n += 1
    st = world.stats
    return dict(winner=world.winner or "imp", reason=world.reason or "timeout", seconds=world.time,
                kills=st["kills"], meetings=st["meetings"], good_ejections=st["good_ejections"],
                wrong_ejections=st["wrong_ejections"], tasks_done=st["tasks_done"],
                witnessed_kills=st["witnessed_kills"], skipped_votes=st["skipped_votes"],
                emergencies=st.get("emergencies", 0))
