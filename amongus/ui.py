"""Briques d'interface communes (polices, texte, bannière, écrans de réunion/vote, fin de partie)."""
from __future__ import annotations

import math
import random
import sys

import pygame

from . import canvas as D
from . import scene
from . import style as ST
from .config import PLAYER_NAMES
from .mapdata import ROOMS

W, H = 1200, 740

BG = (17, 20, 33)
PANEL = (27, 32, 50)
EDGE = (62, 72, 104)
TXT = (230, 233, 245)
DIM = (140, 148, 172)
GOOD = (96, 200, 120)
BAD = (232, 86, 86)
WARN = (240, 180, 70)

COLORS = [(222, 62, 62), (66, 112, 232), (62, 184, 92), (240, 126, 194), (242, 152, 40),
          (236, 216, 72), (72, 212, 222), (150, 110, 230)]

_bg_cache = None


def _draw_meeting_bg(s):
    s.fill(ST.BLUE_BG)
    rng = random.Random(7)
    for _ in range(160):
        c = rng.randint(110, 230)
        D.circle(s, (c, c, min(255, c + 25)), (rng.randrange(W), rng.randrange(H)), rng.choice((1, 1, 2)))


def meeting_background(screen=None):
    """Fond bleu nuit étoilé des réunions (calculé une fois ; version nette enregistrée sur l'écran `screen`)."""
    global _bg_cache
    if _bg_cache is None:
        _bg_cache = pygame.Surface((W, H))
        _draw_meeting_bg(_bg_cache)
    if isinstance(screen, D.Canvas) and id(_bg_cache) not in screen._hires:
        hi = pygame.Surface(screen.R((0, 0, W, H), offset=False).size)
        _draw_meeting_bg(D.Canvas(hi, screen.k, (0, 0), (W, H)))
        screen.register(_bg_cache, hi, screen.k)
    return _bg_cache


# couleurs lisibles sur les panneaux clairs (les couleurs du jeu sont pensées pour un fond sombre)
ON_LIGHT = {BAD: (176, 32, 40), GOOD: (30, 128, 58), WARN: (156, 98, 8), TXT: (22, 26, 36), DIM: (74, 82, 102)}


def on_light(c):
    return ON_LIGHT.get(c, c)


class UIBase:
    """Fonctions d'affichage partagées. Attend : self.screen, self.clock, self.world, self.human, self.log."""

    # ------------------------------------------------------------------ outils
    def font(self, size, bold=False):
        return D.font(size, bold)

    def text(self, s, pos, size=16, color=TXT, bold=False, center=False, right=False):
        anchor = {"center": pos} if center else ({"topright": pos} if right else {"topleft": pos})
        return D.label(self.screen, self.font(size, bold), s, color, outlined=False, **anchor)

    def otext(self, s, pos, size=16, color=(255, 255, 255), bold=True, center=False, right=False, width=2):
        """Texte blanc (ou coloré) cerné de noir, lisible sur tout fond (style cartoon)."""
        anchor = {"center": pos} if center else ({"topright": pos} if right else {"topleft": pos})
        return D.label(self.screen, self.font(size, bold), s, color, width, **anchor)

    def mouse_pos(self):
        p = pygame.mouse.get_pos()
        return self.screen.to_logical(p) if isinstance(self.screen, D.Canvas) else p

    def wrap(self, s, size, width, bold=False):
        words, lines, cur = s.split(), [], ""
        f = self.font(size, bold)
        for w in words:
            t = (cur + " " + w).strip()
            if f.size(t)[0] <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def poll(self):
        evs = []
        if self.inject:
            while self.inject:
                evs.append(self.inject.popleft())
        else:
            evs = pygame.event.get()
        out = []
        for e in evs:
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if isinstance(self.screen, D.Canvas) and e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                                                                 pygame.MOUSEBUTTONUP) and not getattr(e, "logical", 0):
                d = dict(e.dict)
                d["pos"] = self.screen.to_logical(e.pos)
                if "rel" in d:
                    d["rel"] = (e.rel[0] / self.screen.k, e.rel[1] / self.screen.k)
                d["logical"] = 1
                e = pygame.event.Event(e.type, d)
            out.append(e)
        return out

    def name(self, p: int) -> str:
        n = PLAYER_NAMES[p]
        return n + " (vous)" if p == self.human else n

    def add_log(self, msg, color=TXT):
        self.log.append((msg, color))

    def draw_overlay_box(self, x, y, w, h):
        ST.orect(self.screen, ST.PANEL, (x, y, w, h), 14, 5)

    def banner(self, text, sub, ms):
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < ms:
            self.poll()
            self.screen.fill((150, 28, 34))
            k = (pygame.time.get_ticks() - start) / 1000.0
            for i in range(9):                                   # bandes d'alerte
                x = int((i * 170 + k * 260) % (W + 340)) - 170
                D.polygon(self.screen, (176, 38, 44), [(x, 0), (x + 80, 0), (x - 120, H), (x - 200, H)])
            self.otext(text, (W // 2, H // 2 - 24), 58, center=True, width=4)
            self.otext(sub, (W // 2, H // 2 + 38), 22, (255, 225, 225), center=True, width=3)
            pygame.display.flip()
            self.clock.tick(60)

    # -------------------------------------------------------------- réunion / vote
    def evidence_lines(self, ctx):
        """Bulles du panneau : corps, réunion d'urgence, accusations (vraies ou mensongères)."""
        lines = []
        if getattr(ctx, "emergency", False):
            lines.append((f"Réunion d'urgence appelée par {self.name(ctx.reporter)}.", WARN))
        for (room, victim) in ctx.bodies:
            lines.append((f"{PLAYER_NAMES[victim]} est mort(e) : {ROOMS[room]}", BAD))
        for a, t in getattr(ctx, "accuse", {}).items():
            who = "Vous dites" if a == self.human else f"{PLAYER_NAMES[a]} dit"
            lines.append((f"{who} : « J'ai vu {PLAYER_NAMES[t]} tuer ! »", BAD))
        if not getattr(ctx, "accuse", {}):
            lines.append(("Personne ne dit avoir vu le meurtre.", DIM))
        return lines

    def claim_lines(self, ctx):
        """Déclarations « j'étais en ... » et ce qu'en disent les autres -> [(texte, couleur)]."""
        out = []
        checked = bool(getattr(ctx, "confirm", {}))        # recoupements faits (sinon : l'imposteur n'a pas encore parlé)
        for p, room in sorted(getattr(ctx, "claims", {}).items()):
            if not self.world.alive[p]:
                continue
            notes = []
            if not checked:
                out.append((p, f"{self.name(p)} : « J'étais en {ROOMS[room]} »", [], TXT))
                continue
            names = lambda ps: ", ".join(PLAYER_NAMES[c] for c in ps)
            if ctx.contra.get(p):
                notes.append((f"• vu ailleurs par {names(ctx.contra[p])}", BAD))
            if ctx.absent.get(p):
                notes.append((f"• {names(ctx.absent[p])} y étai{'en' if len(ctx.absent[p]) > 1 else ''}t sans l'y voir", BAD))
            if ctx.confirm.get(p):
                notes.append((f"• vu là par {names(ctx.confirm[p])}", GOOD))
            if not notes:
                notes.append(("• personne pour confirmer", WARN))
            col = BAD if (ctx.contra.get(p) or ctx.absent.get(p)) else (GOOD if ctx.confirm.get(p) else WARN)
            out.append((p, f"{self.name(p)} : « J'étais en {ROOMS[room]} »", notes, col))
        return out

    def _card_rects(self, n):
        rects = []
        for i in range(n):
            col, row = i % 2, i // 2
            rects.append(pygame.Rect(48 + col * 368, 128 + row * 106, 352, 92))
        return rects

    def _draw_meeting_frame(self, title, subtitle):
        self.screen.fill(ST.BLUE_BG)
        self.screen.blit(meeting_background(self.screen), (0, 0))
        ST.orect(self.screen, ST.PANEL, (28, 22, W - 56, H - 44), 18, 5)
        self.otext(title, (W // 2, 50), 32, center=True, width=3)
        self.text(subtitle, (W // 2, 86), 15, ST.MUTED, True, center=True)

    def _bubble(self, rect, lines, color, size=14):
        rect = pygame.Rect(rect)
        D.polygon(self.screen, ST.OUTLINE, [(rect.x + 12, rect.bottom), (rect.x + 34, rect.bottom),
                                                      (rect.x + 10, rect.bottom + 13)])
        ST.orect(self.screen, ST.CARD, rect, 10, 3)
        D.polygon(self.screen, ST.CARD, [(rect.x + 15, rect.bottom - 3), (rect.x + 29, rect.bottom - 3),
                                                   (rect.x + 13, rect.bottom + 7)])
        for i, ln in enumerate(lines):
            self.text(ln, (rect.x + 12, rect.y + 7 + 18 * i), size, color, True)

    def _draw_testimony(self, ctx, voter):
        box = pygame.Rect(792, 120, 362, 578)
        ST.orect(self.screen, (226, 232, 242), box, 12, 4)
        self.text("TÉMOIGNAGES", (box.centerx, box.y + 20), 16, ST.INK, True, center=True)
        y = box.y + 42
        for s_, c in self.evidence_lines(ctx):
            lines = self.wrap(s_, 14, box.w - 60, True)
            h = 12 + 18 * len(lines)
            self._bubble((box.x + 14, y, box.w - 28, h), lines, on_light(c))
            y += h + 20
        y += 2
        self.text("OÙ ÉTIEZ-VOUS ?", (box.centerx, y + 8), 15, ST.INK, True, center=True)
        y += 28
        rows = self.claim_lines(ctx)
        if not rows:
            self.text("Pas de meurtre depuis la dernière réunion.", (box.x + 18, y), 13, ST.MUTED, True)
        for p, head, notes, col in rows:
            if y > box.bottom - 30:
                break
            ST.ocircle(self.screen, COLORS[p], (box.x + 24, y + 9), 6, 2)
            self.text(head, (box.x + 38, y), 13, ST.INK, True)
            y += 18
            for txt, c in notes:
                self.text(txt, (box.x + 46, y), 12, on_light(c), True)
                y += 16
            y += 5

    def _draw_card(self, r, p, state, selected=False, hover=False, dots=(), ejected=False, reporter=False):
        """state : 'alive' | 'dead'."""
        dead = state == "dead"
        base = (150, 158, 174) if dead else ST.CARD
        if not dead and selected:
            base = (255, 232, 140)
        elif not dead and hover:
            base = (255, 252, 236)
        ST.orect(self.screen, base, r, 12, 4 if selected else 3)
        scene.draw_bot(self.screen, r.x + 46, r.bottom - 14, COLORS[p], 1, dead=dead)
        self.text(self.name(p), (r.x + 90, r.y + 16), 21, (60, 66, 82) if dead else ST.INK, True)
        if dead:
            self.text("ÉJECTÉ(E)" if ejected else "MORT", (r.x + 90, r.y + 50), 14, (120, 20, 26), True)
        else:
            cleared = getattr(self, "_cur_cleared", set())
            if p == self.human:
                self.text("c'est vous", (r.x + 90, r.y + 50), 14, (38, 92, 190), True)
            elif self.human is not None and self.human in self.world.imps and p in self.world.imps:
                self.text("votre partenaire", (r.x + 90, r.y + 50), 14, (176, 32, 40), True)
            elif p in cleared:
                self.text("alibi confirmé", (r.x + 90, r.y + 50), 14, (30, 128, 58), True)
        if reporter:
            chip = pygame.Rect(r.right - 118, r.y + 9, 106, 22)
            ST.orect(self.screen, ST.RED, chip, 11, 2)
            self.otext("A SIGNALÉ", chip.center, 12, center=True, width=1)
        for i, v in enumerate(dots):
            c = (r.right - 24 - 24 * (i % 7), r.bottom - 22 - 24 * (i // 7))
            ST.ocircle(self.screen, COLORS[v], c, 8, 3)

    def big_button(self, rect, label, fill, enabled=True, hover=False):
        rect = pygame.Rect(rect)
        col = fill if enabled else (170, 178, 192)
        if enabled and hover:
            col = ST.shade(col, 1.08)
        D.rect(self.screen, ST.OUTLINE, rect.move(0, 5).inflate(8, 8), border_radius=14)
        ST.orect(self.screen, col, rect, 10, 4)
        if enabled:
            self.otext(label, rect.center, 18, center=True, width=2)
        else:
            self.text(label, rect.center, 18, (236, 239, 245), True, center=True)

    def meeting_banner(self, ctx):
        """Bannière « réunion » : une seule fois par réunion, avant tout écran de la réunion."""
        if getattr(self, "_bannered", None) is ctx or self.anim_scale <= 0:
            self._bannered = ctx
            return
        self._bannered = ctx
        ms = int((900 if self.human is not None else 500) * self.anim_scale)
        if getattr(ctx, "emergency", False):
            self.banner("RÉUNION D'URGENCE !", f"{PLAYER_NAMES[ctx.reporter]} a appuyé sur le bouton", ms)
        else:
            self.banner("CORPS DÉCOUVERT !", f"Réunion — {PLAYER_NAMES[ctx.reporter]} a signalé un corps", ms)

    def vote_screen(self, ctx, voter, interactive=True, revive=None, auto_ms=None):
        """Écran de vote. Renvoie le joueur choisi, ou None (passer). Non interactif : attend une touche (ou
        `auto_ms`). `revive` : joueur éjecté à l'issue de cette réunion, montré vivant pendant la discussion."""
        w = self.world
        self._cur_cleared = ctx.cleared
        n = w.n
        rects = self._card_rects(n)
        skip = pygame.Rect(48, 128 + 4 * 106 + 14, 240, 52)
        ok = pygame.Rect(48 + 368 - 10, 128 + 4 * 106 + 14, 280, 52)
        sel = None
        start = pygame.time.get_ticks()
        title = "Qui est l'imposteur ?"
        kind = "Réunion d'urgence" if getattr(ctx, "emergency", False) else "Corps signalé"
        sub = (f"{kind} — vous êtes l'IMPOSTEUR : accusez un innocent ou passez." if voter in w.imps else
               f"{kind} — choisissez un joueur puis votez.") if interactive else \
            f"{kind} — vous observez la réunion (une touche pour passer)."
        while True:
            mouse = self.mouse_pos()
            self._draw_meeting_frame(title, sub)
            for p, r in enumerate(rects):
                alive = bool(w.alive[p]) or p == revive
                self._draw_card(r, p, "alive" if alive else "dead", selected=(sel == p),
                                hover=interactive and alive and p != voter and r.collidepoint(mouse),
                                reporter=(p == ctx.reporter))
                if interactive and p == voter:
                    D.rect(self.screen, (38, 92, 190), r.inflate(10, 10), 4, border_radius=16)
            self._draw_testimony(ctx, voter)
            if interactive:
                self.big_button(skip, "PASSER  [S]", (135, 147, 166) if sel != -1 else (96, 108, 130), True,
                                skip.collidepoint(mouse))
                self.big_button(ok, "VOTER  [Entrée]", ST.GREEN, sel is not None, ok.collidepoint(mouse))
            pygame.display.flip()
            self.clock.tick(30)
            if not interactive and auto_ms is not None and pygame.time.get_ticks() - start > auto_ms:
                return None
            for e in self.poll():
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and interactive:
                    for p, r in enumerate(rects):
                        if r.collidepoint(e.pos) and w.alive[p] and p != voter:
                            sel = None if sel == p else p
                    if skip.collidepoint(e.pos):
                        sel = -1 if sel != -1 else None
                    if ok.collidepoint(e.pos) and sel is not None:
                        return None if sel == -1 else sel
                elif e.type == pygame.KEYDOWN:
                    if not interactive:
                        return None
                    if e.key == pygame.K_s:
                        sel = -1
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and sel is not None:
                        return None if sel == -1 else sel
                    elif e.key == pygame.K_ESCAPE:
                        return None
                    elif pygame.K_1 <= e.key <= pygame.K_9:
                        p = e.key - pygame.K_1
                        if p < n and w.alive[p] and p != voter:
                            sel = p

    def human_vote(self, env, voter, ctx):
        self.meeting_banner(ctx)
        return self.vote_screen(ctx, voter, interactive=True)

    def human_claim(self, env, imp, ctx):
        """Imposteur humain : choisir où l'on dit avoir été, et qui accuser. -> (salle, accusé | None)."""
        self.meeting_banner(ctx)
        w = self.world
        room_r = [pygame.Rect(60 + (i % 3) * 236, 168 + (i // 3) * 58, 220, 44) for i in range(len(ROOMS))]
        others = [p for p in range(w.n) if p not in w.imps and w.alive[p]]
        acc_r = [pygame.Rect(60 + (i % 4) * 176, 446 + (i // 4) * 54, 162, 42) for i in range(len(others) + 1)]
        ok = pygame.Rect(60, 600, 320, 54)
        room, acc = None, None
        sub = "Vous êtes l'IMPOSTEUR. Les crewmates ont parlé : où dites-vous avoir été ? Accusez-vous quelqu'un ?"
        while True:
            mouse = self.mouse_pos()
            self._draw_meeting_frame("À vous de parler", sub)
            self.text(f"Le meurtre a eu lieu en {ROOMS[ctx.kill_room]}.   « J'étais en ... »", (60, 130), 17, ST.INK, True)
            for i, r in enumerate(room_r):
                n_claim = sum(1 for q, c in ctx.claims.items() if c == i and q != imp)
                col = (255, 232, 140) if room == i else ((255, 252, 236) if r.collidepoint(mouse) else ST.CARD)
                ST.orect(self.screen, col, r, 10, 4 if room == i else 3)
                if i == ctx.kill_room:
                    D.rect(self.screen, ST.RED, r.inflate(-6, -6), 3, border_radius=8)
                label = ROOMS[i] + (f"  ({n_claim} y étai{'en' if n_claim > 1 else ''}t)" if n_claim else "")
                self.text(label, r.center, 14, ST.INK, True, center=True)
            self.text("Accuser : « J'ai vu ... tuer ! »", (60, 412), 17, ST.INK, True)
            for i, r in enumerate(acc_r):
                p = others[i] if i < len(others) else None
                sel = (acc == p) if p is not None else (acc is None)
                col = (255, 232, 140) if sel else ((255, 252, 236) if r.collidepoint(mouse) else ST.CARD)
                ST.orect(self.screen, col, r, 10, 4 if sel else 3)
                if p is not None:
                    ST.ocircle(self.screen, COLORS[p], (r.x + 22, r.centery), 9, 2)
                self.text(PLAYER_NAMES[p] if p is not None else "Personne", (r.x + 40, r.centery - 9), 15, ST.INK, True)
            self.big_button(ok, "DÉCLARER  [Entrée]", ST.GREEN, room is not None, ok.collidepoint(mouse))
            self._draw_testimony(ctx, imp)
            pygame.display.flip()
            self.clock.tick(30)
            for e in self.poll():
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for i, r in enumerate(room_r):
                        if r.collidepoint(e.pos):
                            room = i
                    for i, r in enumerate(acc_r):
                        if r.collidepoint(e.pos):
                            acc = others[i] if i < len(others) else None
                    if ok.collidepoint(e.pos) and room is not None:
                        return room, acc
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and room is not None:
                    return room, acc

    def show_meeting_result(self, ev):
        """Dépouillement : les votes apparaissent un par un sur les cartes, puis le verdict."""
        w = self.world
        ctx = ev["ctx"]
        self._cur_cleared = ctx.cleared
        self.meeting_banner(ctx)
        spectator = self.human is None
        if spectator or (not w.alive[self.human] and self.human not in ev["votes"]):
            self.vote_screen(ctx, voter=-1, interactive=False, revive=ev["ejected"],
                             auto_ms=int(7000 * self.anim_scale) if spectator else None)
        votes = list(ev["votes"].items())
        rects = self._card_rects(w.n)
        skip_box = pygame.Rect(48, 128 + 4 * 106 + 14, 480, 56)
        fast = self.anim_scale <= 0
        shown = len(votes) if fast else 0
        last = pygame.time.get_ticks()
        done_at = None
        while True:
            self._draw_meeting_frame("Résultat du vote",
                                     "Dépouillement en cours..." if shown < len(votes) else "Tous les votes sont comptés.")
            cur = votes[:shown]
            for p, r in enumerate(rects):
                dead_now = (not w.alive[p]) and not (ev["ejected"] == p and shown < len(votes))
                dots = [v for v, t in cur if t == p]
                self._draw_card(r, p, "dead" if dead_now else "alive", dots=dots,
                                ejected=(ev["ejected"] == p and shown >= len(votes)),
                                reporter=(p == ctx.reporter))
                if ev["ejected"] == p and shown >= len(votes):
                    D.rect(self.screen, ST.RED, r.inflate(10, 10), 5, border_radius=16)
            ST.orect(self.screen, (135, 147, 166), skip_box, 10, 3)
            self.otext("VOTE BLANC", (skip_box.x + 16, skip_box.centery - 11), 17)
            for i, (v, t) in enumerate([(v, t) for v, t in cur if t is None]):
                ST.ocircle(self.screen, COLORS[v], (skip_box.x + 160 + 30 * i, skip_box.centery), 9, 3)
            box = pygame.Rect(792, 120, 362, 578)
            ST.orect(self.screen, (226, 232, 242), box, 12, 4)
            self.text("VOTES", (box.centerx, box.y + 20), 16, ST.INK, True, center=True)
            for i, (v, t) in enumerate(cur):
                y = box.y + 50 + 30 * i
                ST.ocircle(self.screen, COLORS[v], (box.x + 30, y + 10), 8, 2)
                txt = f"{PLAYER_NAMES[v]} : vote blanc" if t is None else f"{PLAYER_NAMES[v]} vote contre {PLAYER_NAMES[t]}"
                self.text(txt, (box.x + 50, y + 1), 15, ST.INK, True)
            if shown >= len(votes):
                yy = box.bottom - 150
                if ev["ejected"] is None:
                    self.text("Personne n'est éjecté.", (box.centerx, yy + 30), 20, ST.INK, True, center=True)
                    self.text("(égalité ou majorité de votes blancs)", (box.centerx, yy + 58), 14, ST.MUTED, True,
                              center=True)
                else:
                    e = ev["ejected"]
                    self.text(f"{PLAYER_NAMES[e]} est éjecté(e).", (box.centerx, yy + 26), 20, ST.INK, True, center=True)
                self.text("Appuyez pour la suite...", (box.centerx, yy + 98), 14, ST.MUTED, True, center=True)
            pygame.display.flip()
            self.clock.tick(30)
            now = pygame.time.get_ticks()
            if shown < len(votes) and now - last > 520:
                shown += 1
                last = now
            if shown >= len(votes) and done_at is None:
                done_at = now
            evs = self.poll()
            if any(e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN) for e in evs):
                if shown < len(votes):
                    shown = len(votes)
                elif now - (done_at or now) > 250 or fast:
                    return
            if fast and shown >= len(votes):
                return
            if self.human is None and done_at is not None and now - done_at > 3500 * self.anim_scale:
                return                                           # spectateur : on enchaîne tout seul

    # ---------------------------------------------------------------- fin de partie
    def end_screen(self):
        env = self.world
        crew_win = env.winner == "crew"
        reason = {"tasks": "Toutes les tâches des crewmates vivants sont terminées.",
                  "ejected": "Les imposteurs ont tous été éjectés.",
                  "parity": f"Il ne reste que {len(env.crew_alive())} crewmate(s) en vie, autant que d'imposteurs.",
                  "timeout": "Temps écoulé."}.get(env.reason, "")
        self.score["crew" if crew_win else "imp"] += 1
        if self.human is None:
            head, hc = ("VICTOIRE DES CREWMATES" if crew_win else "VICTOIRE DES IMPOSTEURS"), (ST.GREEN if crew_win else ST.RED)
        else:
            won = crew_win != self.human_is_imp
            head, hc = ("VOUS AVEZ GAGNÉ !" if won else "VOUS AVEZ PERDU..."), (ST.GREEN if won else ST.RED)
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((6, 9, 20, 150))
        while True:
            self.draw(flip=False)
            self.screen.blit(shade, (0, 0))
            box = pygame.Rect(250, 180, 700, 350)
            self.draw_overlay_box(*box)
            self.otext(head, (600, 236), 40, hc, center=True, width=4)
            if self.human is not None:
                self.text("Victoire des crewmates" if crew_win else "Victoire des imposteurs", (600, 284), 20, ST.INK,
                          True, center=True)
            self.text(reason, (600, 316), 17, ST.MUTED, True, center=True)
            for k, q in enumerate(env.imps):
                scene.draw_bot(self.screen, 600 + (k - (len(env.imps) - 1) / 2) * 70, 410, COLORS[q], 1)
            names = " et ".join(PLAYER_NAMES[q] for q in env.imps)
            self.text(("Les imposteurs étaient : " if len(env.imps) > 1 else "L'imposteur était : ") + names,
                      (600, 428), 19, (176, 32, 40), True, center=True)
            st = env.stats
            self.text(f"{st['kills']} kill(s) · {st['meetings']} réunion(s) · {st['tasks_done']} tâche(s) faite(s)"
                      f"   —   score : crewmates {self.score['crew']} · imposteurs {self.score['imp']}",
                      (600, 458), 14, ST.MUTED, True, center=True)
            self.otext("R : rejouer      M : menu      Échap : quitter" if getattr(self, "menu", False)
                       else "R : rejouer      Échap : quitter", (600, 500), 18, center=True)
            pygame.display.flip()
            self.clock.tick(30)
            for e in self.poll():
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        return True
                    if e.key == pygame.K_m and getattr(self, "menu", False):
                        return "menu"
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return False
