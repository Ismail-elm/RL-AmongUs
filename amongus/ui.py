"""Briques d'interface communes (polices, texte, bannière, écrans de réunion/vote, fin de partie)."""
from __future__ import annotations

import math
import random
import sys

import pygame

from . import scene
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


def meeting_background():
    """Fond dégradé bleu nuit avec étoiles (calculé une fois)."""
    global _bg_cache
    if _bg_cache is None:
        s = pygame.Surface((W, H))
        for y in range(H):
            k = y / H
            pygame.draw.line(s, (int(10 + 14 * k), int(14 + 20 * k), int(30 + 34 * k)), (0, y), (W, y))
        rng = random.Random(7)
        for _ in range(140):
            c = rng.randint(90, 210)
            pygame.draw.circle(s, (c, c, min(255, c + 30)), (rng.randrange(W), rng.randrange(H)), rng.choice((1, 1, 2)))
        _bg_cache = s
    return _bg_cache


class UIBase:
    """Fonctions d'affichage partagées. Attend : self.screen, self.clock, self.world, self.human, self.log."""

    # ------------------------------------------------------------------ outils
    def font(self, size, bold=False):
        k = (size, bold)
        if k not in self.fonts:
            self.fonts[k] = pygame.font.SysFont("dejavusans,arial,helvetica", size, bold=bold)
        return self.fonts[k]

    def text(self, s, pos, size=16, color=TXT, bold=False, center=False, right=False):
        surf = self.font(size, bold).render(s, True, color)
        r = surf.get_rect()
        if center:
            r.center = pos
        elif right:
            r.topright = pos
        else:
            r.topleft = pos
        self.screen.blit(surf, r)
        return r

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
        for e in evs:
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        return evs

    def name(self, p: int) -> str:
        n = PLAYER_NAMES[p]
        return n + " (vous)" if p == self.human else n

    def add_log(self, msg, color=TXT):
        self.log.append((msg, color))

    def draw_overlay_box(self, x, y, w, h):
        r = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, PANEL, r, border_radius=14)
        pygame.draw.rect(self.screen, EDGE, r, 2, border_radius=14)

    def banner(self, text, sub, ms):
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < ms:
            self.poll()
            self.screen.fill((70, 12, 14))
            self.text(text, (W // 2, H // 2 - 20), 54, (255, 226, 226), True, center=True)
            self.text(sub, (W // 2, H // 2 + 36), 22, (255, 170, 170), center=True)
            pygame.display.flip()
            self.clock.tick(60)

    # -------------------------------------------------------------- réunion / vote
    def evidence_lines(self, ctx):
        lines = []
        for (room, victim) in ctx.bodies:
            lines.append((f"Corps de {PLAYER_NAMES[victim]} trouvé : {ROOMS[room]}", BAD))
        for w in ctx.witnesses:
            who = "Vous avez" if w == self.human else f"{PLAYER_NAMES[w]} a"
            lines.append((f"{who} VU {PLAYER_NAMES[ctx.imposter]} commettre le meurtre !", BAD))
        if not ctx.witnesses:
            lines.append(("Aucun témoin direct du meurtre.", DIM))
        return lines

    def _card_rects(self, n):
        rects = []
        for i in range(n):
            col, row = i % 2, i // 2
            rects.append(pygame.Rect(40 + col * 372, 132 + row * 108, 356, 94))
        return rects

    def _draw_meeting_frame(self, title, subtitle):
        self.screen.blit(meeting_background(), (0, 0))
        pan = pygame.Rect(24, 20, W - 48, H - 40)
        pygame.draw.rect(self.screen, (22, 30, 52), pan, border_radius=22)
        pygame.draw.rect(self.screen, (120, 150, 200), pan, 3, border_radius=22)
        pygame.draw.rect(self.screen, (40, 54, 92), (24, 20, W - 48, 78), border_top_left_radius=22,
                         border_top_right_radius=22)
        self.text(title, (W // 2, 46), 30, (255, 255, 255), True, center=True)
        self.text(subtitle, (W // 2, 78), 15, (170, 190, 230), center=True)

    def _draw_testimony(self, ctx, voter):
        box = pygame.Rect(790, 118, 370, 590)
        pygame.draw.rect(self.screen, (16, 22, 40), box, border_radius=16)
        pygame.draw.rect(self.screen, (80, 100, 150), box, 2, border_radius=16)
        self.text("TÉMOIGNAGES", (box.centerx, box.y + 22), 16, (170, 190, 230), True, center=True)
        y = box.y + 50
        for s, c in self.evidence_lines(ctx):
            lines = self.wrap(s, 14, box.w - 70, True)
            h = 16 + 19 * len(lines)
            b = pygame.Rect(box.x + 14, y, box.w - 28, h)
            pygame.draw.rect(self.screen, (236, 240, 250), b, border_radius=12)
            pygame.draw.polygon(self.screen, (236, 240, 250), [(b.x + 14, b.bottom - 2), (b.x + 30, b.bottom - 2), (b.x + 12, b.bottom + 9)])
            dark = {BAD: (170, 30, 40), DIM: (80, 86, 108)}.get(c, (30, 36, 60))
            for i, ln in enumerate(lines):
                self.text(ln, (b.x + 12, b.y + 8 + 19 * i), 14, dark, True)
            y += h + 18
        # indices d'alibi
        y += 4
        self.text("ALIBIS", (box.centerx, y + 8), 14, (170, 190, 230), True, center=True)
        y += 28
        w = self.world
        for p in range(w.n):
            if not w.alive[p] or p == voter:
                continue
            ok = p in ctx.cleared
            pygame.draw.circle(self.screen, GOOD if ok else WARN, (box.x + 30, y + 9), 6)
            self.text(f"{PLAYER_NAMES[p]} : " + ("innocenté(e)" if ok else "non innocenté(e)"), (box.x + 46, y), 14,
                      GOOD if ok else WARN)
            y += 22

    def _draw_card(self, r, p, state, selected=False, hover=False, dots=(), ejected=False, reporter=False):
        """state : 'alive' | 'dead'."""
        dead = state == "dead"
        base = (50, 58, 84) if dead else (232, 238, 250)
        if not dead and (selected or hover):
            base = (255, 244, 190) if selected else (246, 250, 255)
        pygame.draw.rect(self.screen, base, r, border_radius=14)
        pygame.draw.rect(self.screen, (255, 200, 60) if selected else (110, 130, 170), r, 4 if selected else 2, border_radius=14)
        scene.draw_bot(self.screen, r.x + 48, r.bottom - 14, COLORS[p], 1, dead=dead)
        tc = (110, 118, 142) if dead else (22, 30, 52)
        self.text(self.name(p), (r.x + 92, r.y + 16), 20, tc, True)
        if dead:
            self.text("MORT" if not ejected else "ÉJECTÉ(E)", (r.x + 92, r.y + 50), 14, (200, 90, 90), True)
        else:
            cleared = getattr(self, "_cur_cleared", set())
            if p == self.human:
                self.text("c'est vous", (r.x + 92, r.y + 50), 14, (60, 110, 200), True)
            elif p in cleared:
                self.text("✔ innocenté(e)", (r.x + 92, r.y + 50), 14, (30, 140, 70), True)
            else:
                self.text("non innocenté(e)", (r.x + 92, r.y + 50), 14, (190, 120, 20), True)
        if reporter:
            pygame.draw.rect(self.screen, (226, 84, 70), (r.right - 118, r.y + 8, 110, 22), border_radius=11)
            self.text("A SIGNALÉ", (r.right - 63, r.y + 19), 12, (255, 255, 255), True, center=True)
        for i, v in enumerate(dots):
            c = (r.right - 24 - 22 * (i % 7), r.bottom - 22 - 22 * (i // 7))
            pygame.draw.circle(self.screen, (255, 255, 255), c, 11)
            pygame.draw.circle(self.screen, COLORS[v], c, 8)

    def vote_screen(self, ctx, voter, interactive=True):
        """Écran de vote. Renvoie le joueur choisi, ou None (skip). Non interactif : attend une touche."""
        w = self.world
        self._cur_cleared = ctx.cleared
        n = w.n
        rects = self._card_rects(n)
        skip = pygame.Rect(40, 132 + 4 * 108 + 6, 230, 50)
        ok = pygame.Rect(40 + 372 - 20, 132 + 4 * 108 + 6, 270, 50)
        sel = None
        sub = ("Vous êtes l'IMPOSTEUR : accusez un innocent ou passez." if w.imp == voter else
               "Qui est l'imposteur ? Choisissez un joueur puis confirmez.") if interactive else \
            "Vous observez la réunion — appuyez sur une touche."
        while True:
            mouse = pygame.mouse.get_pos()
            self._draw_meeting_frame("RÉUNION D'URGENCE", sub)
            for p, r in enumerate(rects):
                alive = bool(w.alive[p])
                self._draw_card(r, p, "alive" if alive else "dead", selected=(sel == p),
                                hover=interactive and alive and p != voter and r.collidepoint(mouse),
                                reporter=(p == ctx.reporter))
                if interactive and p == voter:
                    pygame.draw.rect(self.screen, (22, 30, 52), r.inflate(-4, -4), border_radius=12)
                    self._draw_card(r, p, "alive", reporter=(p == ctx.reporter))
                    pygame.draw.rect(self.screen, (60, 120, 220), r, 4, border_radius=14)
            self._draw_testimony(ctx, voter)
            if interactive:
                sk_sel = sel == -1
                pygame.draw.rect(self.screen, (255, 244, 190) if sk_sel else ((200, 210, 232) if skip.collidepoint(mouse) else (150, 164, 196)),
                                 skip, border_radius=12)
                self.text("PASSER (skip)  [S]", skip.center, 17, (22, 30, 52), True, center=True)
                pygame.draw.rect(self.screen, (60, 190, 100) if sel is not None else (60, 74, 100), ok, border_radius=12)
                self.text("CONFIRMER  [Entrée]", ok.center, 18, (255, 255, 255) if sel is not None else (120, 132, 160), True, center=True)
            pygame.display.flip()
            self.clock.tick(30)
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
        return self.vote_screen(ctx, voter, interactive=True)

    def show_meeting_result(self, ev):
        """Dépouillement : les votes apparaissent un par un sur les cartes, puis le verdict."""
        w = self.world
        ctx = ev["ctx"]
        self._cur_cleared = ctx.cleared
        if self.human is None or (not w.alive[self.human] and self.human not in ev["votes"]):
            self.vote_screen(ctx, voter=-1, interactive=False)
        votes = list(ev["votes"].items())
        rects = self._card_rects(w.n)
        skip_box = pygame.Rect(40, 132 + 4 * 108 + 6, 470, 56)
        fast = self.anim_scale <= 0
        shown = len(votes) if fast else 0
        last = pygame.time.get_ticks()
        done_at = None
        while True:
            self._draw_meeting_frame("RÉSULTAT DU VOTE",
                                     "Dépouillement en cours..." if shown < len(votes) else "Tous les votes sont comptés.")
            cur = votes[:shown]
            for p, r in enumerate(rects):
                dead_now = (not w.alive[p]) and not (ev["ejected"] == p and shown < len(votes))
                dots = [v for v, t in cur if t == p]
                self._draw_card(r, p, "dead" if dead_now else "alive", dots=dots,
                                ejected=(ev["ejected"] == p and shown >= len(votes)),
                                reporter=(p == ctx.reporter))
                if ev["ejected"] == p and shown >= len(votes):
                    pygame.draw.rect(self.screen, (230, 60, 60), r, 5, border_radius=14)
            pygame.draw.rect(self.screen, (150, 164, 196), skip_box, border_radius=12)
            self.text("SKIP", (skip_box.x + 20, skip_box.centery - 11), 18, (22, 30, 52), True)
            for i, (v, t) in enumerate([(v, t) for v, t in cur if t is None]):
                c = (skip_box.x + 120 + 30 * i, skip_box.centery)
                pygame.draw.circle(self.screen, (255, 255, 255), c, 12)
                pygame.draw.circle(self.screen, COLORS[v], c, 9)
            # panneau de verdict à droite
            box = pygame.Rect(790, 118, 370, 590)
            pygame.draw.rect(self.screen, (16, 22, 40), box, border_radius=16)
            pygame.draw.rect(self.screen, (80, 100, 150), box, 2, border_radius=16)
            self.text("VOTES", (box.centerx, box.y + 22), 16, (170, 190, 230), True, center=True)
            for i, (v, t) in enumerate(cur):
                y = box.y + 52 + 30 * i
                scene.draw_bot(self.screen, box.x + 34, y + 22, COLORS[v], 1)
                tgt = "skip" if t is None else PLAYER_NAMES[t]
                self.text(f"{PLAYER_NAMES[v]}  →  {tgt}", (box.x + 62, y + 2), 15, TXT)
            if shown >= len(votes):
                yy = box.bottom - 150
                if ev["ejected"] is None:
                    self.text("Personne n'est éjecté.", (box.centerx, yy + 30), 20, TXT, True, center=True)
                    self.text("(égalité ou majorité de skip)", (box.centerx, yy + 58), 14, DIM, center=True)
                else:
                    e = ev["ejected"]
                    self.text(f"{PLAYER_NAMES[e]} est éjecté(e).", (box.centerx, yy + 26), 20, TXT, True, center=True)
                    self.text("Appuyez pour la suite...", (box.centerx, yy + 70), 14, DIM, center=True)
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

    # ---------------------------------------------------------------- fin de partie
    def end_screen(self):
        env = self.world
        crew_win = env.winner == "crew"
        reason = {"tasks": "Toutes les tâches des crewmates vivants sont terminées.",
                  "ejected": "L'imposteur a été éjecté.",
                  "parity": f"Il ne reste que {len(env.crew_alive())} crewmates en vie.",
                  "timeout": "Temps écoulé."}.get(env.reason, "")
        self.score["crew" if crew_win else "imp"] += 1
        if self.human is None:
            head, hc = ("VICTOIRE DES CREWMATES" if crew_win else "VICTOIRE DE L'IMPOSTEUR"), (GOOD if crew_win else BAD)
        else:
            won = crew_win != self.human_is_imp
            head, hc = ("VOUS AVEZ GAGNÉ !" if won else "VOUS AVEZ PERDU..."), (GOOD if won else BAD)
        while True:
            self.draw()
            self.draw_overlay_box(250, 190, 700, 330)
            self.text(head, (600, 245), 38, hc, True, center=True)
            self.text("Victoire des crewmates" if crew_win else "Victoire de l'imposteur", (600, 290), 20, TXT, center=True)
            self.text(reason, (600, 322), 17, DIM, center=True)
            self.text(f"L'imposteur était : {PLAYER_NAMES[env.imp]}", (600, 362), 19, BAD, True, center=True)
            st = env.stats
            self.text(f"{st['kills']} kill(s) · {st['meetings']} réunion(s) · {st['tasks_done']} tâche(s) faite(s)",
                      (600, 400), 16, DIM, center=True)
            self.text(f"Score cumulé — crewmates {self.score['crew']} : {self.score['imp']} imposteur",
                      (600, 430), 15, DIM, center=True)
            self.text("R : rejouer      Échap : quitter", (600, 480), 18, TXT, True, center=True)
            pygame.display.flip()
            self.clock.tick(30)
            for e in self.poll():
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_r:
                        return True
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return False
