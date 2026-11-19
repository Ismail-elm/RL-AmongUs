"""Moteur de jeu : Among Us simplifié, tour par tour (tous les joueurs jouent en même temps).

Règles (résumé)
---------------
* 7 joueurs : 1 imposteur + 6 crewmates, tous dans la Cafétéria au départ.
* Chaque tour, chaque joueur vivant choisit UNE action :
    - aller dans une salle voisine
    - attendre
    - TÂCHE (crewmate) : il faut `task_duration` tours consécutifs dans la salle de la tâche
    - KILL (imposteur) : tue un crewmate présent dans la même salle (cooldown)
* Un cadavre reste dans sa salle. Dès qu'un crewmate vivant termine un tour dans la
  salle d'un cadavre, il le signale -> RÉUNION : tout le monde revient en Cafétéria, on vote.
* Vote : les crewmates mettent en commun ce qu'ils ont vu. Quelqu'un vu AILLEURS que sur
  le lieu du crime au moment du meurtre est innocenté. Un témoin direct identifie l'imposteur.
  Les crewmates ne votent que s'ils ont peu de suspects (sinon « skip »).
* Victoire crewmates : imposteur éjecté, ou tous les crewmates VIVANTS ont fini leurs tâches.
* Victoire imposteur : il reste <= `crew_left_to_lose` crewmates (3 par défaut), ou temps limite dépassé.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import Config
from .mapdata import ADJ, DIST, NEXT_HOP, N_ROOMS, ROOMS

# --- actions -------------------------------------------------------------
# 0..N_ROOMS-1 : aller dans la salle j (si voisine)
WAIT = N_ROOMS
TASK = N_ROOMS + 1
KILL = N_ROOMS + 2
REPORT = N_ROOMS + 3     # imposteur (monde temps réel) : signaler lui-même un corps, comme dans le vrai jeu
EMERGENCY = N_ROOMS + 4  # monde temps réel : bouton de réunion d'urgence (Cafétéria, une fois par joueur)
VENT_A = N_ROOMS + 5     # imposteur : conduit vers la 1re salle reliée à la grille de sa salle
VENT_B = N_ROOMS + 6     # imposteur : conduit vers la 2e salle reliée
SABOTAGE = N_ROOMS + 7   # imposteur : couper les lumières
FIX = N_ROOMS + 8        # crewmate : réparer les lumières (panneau d'Électricité)
N_ACTIONS = N_ROOMS + 9
NONE_HINT = N_ROOMS  # « pas de direction »


def action_name(a: int) -> str:
    if a < N_ROOMS:
        return f"aller {ROOMS[a]}"
    return {WAIT: "attendre", TASK: "tâche", KILL: "KILL", REPORT: "signaler", EMERGENCY: "réunion d'urgence",
            VENT_A: "conduit A", VENT_B: "conduit B", SABOTAGE: "saboter les lumières", FIX: "réparer les lumières"}[a]


@dataclass
class Kill:
    idx: int            # indice dans le journal des positions
    room: int
    victim: int
    witnesses: list[int]
    # monde temps réel : instantané au moment du meurtre (pour recouper les déclarations en réunion)
    t: float = 0.0
    zone: dict = field(default_factory=dict)     # joueur -> salle où il était (couloir : salle la plus proche)
    rooms: dict = field(default_factory=dict)    # joueur -> salle exacte (-1 : couloir)
    sight: dict = field(default_factory=dict)    # joueur -> ensemble des joueurs qu'il voyait
    dark: bool = False                           # lumières coupées au moment du meurtre
    killer: int = -1                             # qui a tué
    killer: int = -1                             # qui a tué


@dataclass
class MeetingCtx:
    reporter: int
    bodies: list[tuple[int, int]]          # (salle, victime)
    cleared: set[int]                       # joueurs innocentés par des témoignages
    witnesses: list[int]                    # crewmates vivants ayant vu le meurtre
    imposter: int                           # (usage interne : témoignages honnêtes) premier imposteur
    votes: dict = field(default_factory=dict)
    # monde temps réel : discussion
    emergency: bool = False                 # réunion d'urgence (bouton) plutôt que corps signalé
    kill_room: int = -1                     # salle du meurtre de référence (-1 : aucun meurtre depuis la réunion)
    ref: object = None                      # ce meurtre (Kill), avec l'instantané de qui était où
    shared: object = None                   # crewmate -> joueurs dont il parle (mémoire imparfaite) ; None : tous
    claims: dict = field(default_factory=dict)    # joueur -> salle où il DIT avoir été au moment du meurtre
    accuse: dict = field(default_factory=dict)    # joueur -> joueur qu'il accuse (« je l'ai vu tuer »)
    confirm: dict = field(default_factory=dict)   # joueur -> crewmates qui l'ont vu là où il le dit
    contra: dict = field(default_factory=dict)    # joueur -> crewmates qui l'ont vu AILLEURS
    absent: dict = field(default_factory=dict)    # joueur -> crewmates présents dans cette salle, qui ne l'y ont pas vu


class AmongUsEnv:
    def __init__(self, cfg: Config | None = None, seed: int | None = None):
        self.cfg = cfg or Config()
        self.n_imp = self._n_imp()
        self.n = self.cfg.n_crew + self.n_imp
        self.rng = random.Random(seed)
        # callbacks de vote pour des joueurs « externes » (humain) : f(env, joueur, ctx) -> joueur | None
        self.vote_callbacks: dict[int, callable] = {}
        self.reset()

    def _n_imp(self) -> int:
        return 1                    # jeu tour par tour : un seul imposteur

    # ------------------------------------------------------------------ reset
    def reset(self, imposter=None) -> "AmongUsEnv":
        """imposter : None (tirage), un joueur (les autres imposteurs sont tirés au sort) ou une liste de joueurs."""
        n, cfg = self.n, self.cfg
        if imposter is None:
            imps = self.rng.sample(range(n), self.n_imp)
        elif isinstance(imposter, int):
            imps = [imposter] + self.rng.sample([p for p in range(n) if p != imposter], self.n_imp - 1)
        else:
            imps = list(imposter)
        self.imps = sorted(imps)
        self.imp = imps[0]          # « l'imposteur » quand il n'y en a qu'un
        self.room = [0] * n
        self.alive = [True] * n
        self.tasks: list[set[int]] = [set() for _ in range(n)]
        self.progress = [0] * n
        for p in range(n):
            if p not in self.imps:
                self.tasks[p] = set(self.rng.sample(range(1, N_ROOMS), cfg.tasks_per_crew))
        self.cooldown = cfg.kill_cooldown
        self.bodies: list[tuple[int, int]] = []
        self.kills: list[Kill] = []
        self.log: list[tuple[int, ...]] = []   # positions de tous (post-tour), -1 = mort
        self.imp_mem: int | None = None        # dernière salle où l'imposteur a vu des crewmates
        self.t = 0
        self.winner: str | None = None
        self.reason: str | None = None
        self.events: list[dict] = []
        self.kill_target: int | None = None   # victime choisie par un imposteur humain (sinon au hasard)
        self.stats = dict(kills=0, meetings=0, good_ejections=0, wrong_ejections=0,
                          tasks_done=0, witnessed_kills=0, skipped_votes=0)
        return self

    # --------------------------------------------------------------- requêtes
    def is_imp(self, p) -> bool:
        return p in self.imps

    def crew_alive(self) -> list[int]:
        return [p for p in range(self.n) if p not in self.imps and self.alive[p]]

    def imps_alive(self) -> list[int]:
        return [p for p in self.imps if self.alive[p]]

    def alive_players(self) -> list[int]:
        return [p for p in range(self.n) if self.alive[p]]

    def others_in_room(self, p: int) -> list[int]:
        r = self.room[p]
        return [q for q in range(self.n) if q != p and self.alive[q] and self.room[q] == r]

    def legal_actions(self, p: int) -> list[int]:
        r = self.room[p]
        acts = list(ADJ[r])
        acts.append(WAIT)
        if p == self.imp:
            if self.cooldown == 0 and any(
                q != p and self.alive[q] and self.room[q] == r for q in range(self.n)
            ):
                acts.append(KILL)
        elif r in self.tasks[p]:
            acts.append(TASK)
        return acts

    def _is_legal(self, p: int, a: int) -> bool:
        if a < N_ROOMS:
            return a in ADJ[self.room[p]]
        if a == WAIT:
            return True
        return a in self.legal_actions(p)

    # ------------------------------------------------------------ observations
    def crew_hint(self, p: int) -> int:
        """Salle (voisine) à rejoindre pour la tâche la plus proche ; salle courante si tâche ici."""
        r, ts = self.room[p], self.tasks[p]
        if not ts:
            return NONE_HINT
        if r in ts:
            return r
        best = min(ts, key=lambda t: (DIST[r][t], t))
        return NEXT_HOP[r][best]

    def imp_hint(self, p: int) -> int:
        r, mem = self.room[p], self.imp_mem
        if mem is None:
            return NONE_HINT
        if mem == r:
            return r
        return NEXT_HOP[r][mem]

    # -------------------------------------------------------------------- step
    def step(self, actions: dict[int, int]):
        """Fait jouer un tour. `actions` : {joueur: action}. Renvoie (rewards, terminal, done).

        rewards[p] : récompense de p ; terminal[p] : True si la trajectoire de p s'arrête
        (mort, éjection ou fin de partie). Les événements du tour sont dans self.events.
        """
        cfg, n, imp = self.cfg, self.n, self.imp
        rew = [0.0] * n
        term = [False] * n
        self.events = []
        self.t += 1

        acts: dict[int, int] = {}
        for p in range(n):
            if not self.alive[p]:
                continue
            a = actions.get(p, WAIT)
            if not self._is_legal(p, a):
                a = WAIT
            acts[p] = a
            rew[p] += cfg.r_step

        # 1) déplacements simultanés
        for p, a in acts.items():
            if a < N_ROOMS:
                self.room[p] = a

        # 2) tâches
        for p, a in acts.items():
            if p == imp:
                continue
            if a == TASK:
                self.progress[p] += 1
                rew[p] += cfg.r_task_progress
                if self.progress[p] >= cfg.task_duration:
                    r = self.room[p]
                    self.tasks[p].discard(r)
                    self.progress[p] = 0
                    rew[p] += cfg.r_task
                    self.stats["tasks_done"] += 1
                    self.events.append(dict(type="task", player=p, room=r))
            else:
                self.progress[p] = 0

        # 3) kill
        killed = False
        if acts.get(imp) == KILL:
            r = self.room[imp]
            victims = [q for q in acts if q != imp and self.alive[q] and self.room[q] == r]
            if victims:
                v = self.kill_target if self.kill_target in victims else self.rng.choice(victims)
                witnesses = [q for q in victims if q != v]
                self.alive[v] = False
                self.progress[v] = 0
                rew[v] += cfg.r_death
                term[v] = True
                rew[imp] += cfg.r_kill
                self.kills.append(Kill(idx=len(self.log), room=r, victim=v, witnesses=witnesses))
                self.bodies.append((r, v))
                self.cooldown = cfg.kill_cooldown
                self.stats["kills"] += 1
                if witnesses:
                    self.stats["witnessed_kills"] += 1
                self.events.append(dict(type="kill", killer=imp, victim=v, room=r, witnesses=witnesses))
                killed = True
        if not killed and self.cooldown > 0:
            self.cooldown -= 1
        self.kill_target = None

        # 4) journal des positions + mémoire de l'imposteur
        self.log.append(tuple(self.room[p] if self.alive[p] else -1 for p in range(n)))
        if self.alive[imp]:
            r = self.room[imp]
            if any(q != imp and self.alive[q] and self.room[q] == r for q in range(n)):
                self.imp_mem = r
            elif self.imp_mem == r:
                self.imp_mem = None

        # 5) fin de partie immédiate ? (parité, tâches, temps)
        out = self._outcome()
        if out is None:
            # 6) un cadavre est-il découvert ?
            reporter = None
            for broom, _victim in self.bodies:
                for q in self.crew_alive():
                    if self.room[q] == broom:
                        reporter = q
                        break
                if reporter is not None:
                    break
            if reporter is not None:
                self._meeting(reporter, rew, term)
                out = self._outcome()
        if out is not None:
            self._finish(out[0], out[1], rew, term)
        return rew, term, self.winner is not None

    # ------------------------------------------------------------- fin de partie
    def _parity(self, ac) -> bool:
        return len(ac) <= max(len(self.imps_alive()), self.cfg.crew_left_to_lose)

    def _outcome(self):
        if not self.imps_alive():
            return "crew", "ejected"
        ac = self.crew_alive()
        if self._parity(ac):
            return "imp", "parity"
        if all(not self.tasks[p] for p in ac):
            return "crew", "tasks"
        if self.t >= self.cfg.max_steps:
            return "imp", "timeout"
        return None

    def _finish(self, winner: str, reason: str, rew, term):
        cfg = self.cfg
        self.winner, self.reason = winner, reason
        for p in range(self.n):
            if term[p]:
                continue
            if self.is_imp(p):
                rew[p] += cfg.r_imp_win if winner == "imp" else cfg.r_imp_loss
            elif self.alive[p]:
                rew[p] += cfg.r_crew_win if winner == "crew" else cfg.r_crew_loss
            term[p] = True
        self.events.append(dict(type="end", winner=winner, reason=reason))

    # ----------------------------------------------------------------- réunion
    def _evidence(self):
        """Témoignages honnêtes des crewmates vivants.

        Un joueur vu dans une autre salle que celle du crime, au moment du crime, par un autre
        crewmate, est innocenté. Un témoin direct (présent dans la salle du crime) connaît le tueur.
        """
        n, imp, recall = self.n, self.imp, self.cfg.recall
        crew = self.crew_alive()
        cleared: set[int] = set()
        witnesses: list[int] = []
        for k in self.kills:
            ref = self.log[k.idx]
            for c in crew:
                rc = ref[c]
                if rc < 0 or rc == k.room:
                    continue
                for p in range(n):
                    if p != c and ref[p] == rc and p not in cleared and self.rng.random() < recall:
                        cleared.add(p)
            for w in k.witnesses:
                if self.alive[w] and w != imp and w not in witnesses:
                    witnesses.append(w)
        return cleared, witnesses

    def _prepare_meeting(self, ctx: MeetingCtx):
        """Avant le vote (monde temps réel : déclarations et recoupements). Rien dans le jeu tour par tour."""

    def ai_vote(self, voter: int, ctx: MeetingCtx):
        """Vote heuristique (non appris) des joueurs contrôlés par l'IA."""
        alive = self.alive_players()
        if voter != self.imp:
            if ctx.witnesses:
                return self.imp
            suspects = [p for p in alive if p != voter and p not in ctx.cleared]
            if not suspects or len(suspects) > self.cfg.skip_threshold:
                return None
            return self.rng.choice(suspects)
        # l'imposteur accuse un crewmate non innocenté (pour brouiller les pistes)
        cands = [p for p in alive if p != self.imp and p not in ctx.cleared]
        return self.rng.choice(cands) if cands else None

    def _meeting(self, reporter: int, rew, term):
        cfg = self.cfg
        cleared, witnesses = self._evidence()
        ctx = MeetingCtx(reporter=reporter, bodies=list(self.bodies), cleared=cleared,
                         witnesses=witnesses, imposter=self.imp)
        ctx.imposters = list(self.imps)
        ctx.imposters = list(self.imps)
        self._prepare_meeting(ctx)
        cleared = ctx.cleared
        votes: dict[int, int | None] = {}
        for v in self.alive_players():
            cb = self.vote_callbacks.get(v)
            choice = cb(self, v, ctx) if cb is not None else self.ai_vote(v, ctx)
            if choice is not None and (choice == v or not self.alive[choice]):
                choice = None
            votes[v] = choice
        ctx.votes = votes
        tally: dict[int | None, int] = {}
        for c in votes.values():
            tally[c] = tally.get(c, 0) + 1
        top = max(tally.values())
        leaders = [c for c, k in tally.items() if k == top]
        ejected = leaders[0] if len(leaders) == 1 else None   # égalité ou skip majoritaire -> personne
        self.stats["meetings"] += 1
        self.stats["skipped_votes"] += sum(1 for c in votes.values() if c is None)
        if ejected is not None:
            self.alive[ejected] = False
            term[ejected] = True
            if self.is_imp(ejected):
                rew[ejected] += cfg.r_imp_loss
                self.stats["good_ejections"] += 1
            else:
                rew[ejected] += cfg.r_death
                self.stats["wrong_ejections"] += 1
        self.events.append(dict(type="meeting", reporter=reporter, bodies=list(self.bodies),
                                votes=votes, ejected=ejected,
                                was_imposter=self.is_imp(ejected) if ejected is not None else None,
                                cleared=sorted(cleared), witnesses=list(witnesses), ctx=ctx,
                                emergency=ctx.emergency))
        # retour en Cafétéria
        self.bodies = []
        self.kills = []
        self.log = []
        self.imp_mem = None
        self.cooldown = cfg.kill_cooldown
        for p in range(self.n):
            self.progress[p] = 0
            if self.alive[p]:
                self.room[p] = 0
