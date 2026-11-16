"""Interface TEMPS RÉEL : déplacement libre (ZQSD / WASD / flèches), caméra, vision limitée, boutons."""
from __future__ import annotations

import math
import random
import sys
from collections import deque

import numpy as np
import pygame

from . import rt_map as M
from . import scene
from .config import Config, PLAYER_NAMES
from .ui import (UIBase, BAD, BG, COLORS, DIM, GOOD, PANEL, TXT, WARN, H, W)
from .mapdata import ROOMS
from .rt_world import RTConfig, RTWorld

LIGHT_R = 470
TICK = 1 / 30

_world_surface = None
_mini_cache: dict = {}


# ------------------------------------------------------------------ rendu statique
def _prop_scaled(surf, kind, cx, cy, opt, s=0.7):
    tmp = pygame.Surface((360, 360), pygame.SRCALPHA)
    scene._prop(tmp, kind, 180, 180, opt)
    tmp = pygame.transform.smoothscale(tmp, (int(360 * s), int(360 * s)))
    surf.blit(tmp, tmp.get_rect(center=(cx, cy)))


WALL_H = M.WALL_H


def _vgrad(surf, rect, c_top, c_bot):
    x, y, w, h = rect
    for i in range(h):
        k = i / max(1, h - 1)
        c = tuple(int(c_top[j] + (c_bot[j] - c_top[j]) * k) for j in range(3))
        pygame.draw.line(surf, c, (x, y + i), (x + w - 1, y + i))


def _wall_face(surf, x, y, w, h):
    """Mur du fond vu de face : dessus clair, face dégradée, joints verticaux, rivets, ombre au pied."""
    _vgrad(surf, (x, y, w, h), (74, 84, 112), (36, 42, 62))
    pygame.draw.rect(surf, (138, 152, 184), (x, y, w, 7))
    pygame.draw.rect(surf, (176, 190, 220), (x, y, w, 2))
    for sx in range(x + 40, x + w - 10, 96):
        pygame.draw.line(surf, (30, 36, 54), (sx, y + 9), (sx, y + h), 3)
        pygame.draw.line(surf, (96, 108, 140), (sx + 3, y + 9), (sx + 3, y + h), 1)
        for ry in (y + 18, y + h - 12):
            pygame.draw.circle(surf, (24, 28, 42), (sx - 12, ry), 3)
            pygame.draw.circle(surf, (120, 132, 164), (sx - 12, ry - 1), 1)
    pygame.draw.line(surf, (20, 24, 36), (x, y + h - 1), (x + w, y + h - 1), 2)


def _inner_shadow(surf, rect, top=70, other=34, a=120):
    """Ombre portée douce à l'intérieur du sol (fait ressortir les murs)."""
    x, y, w, h = rect
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    xs = np.arange(w)[None, :]
    ys = np.arange(h)[:, None]
    d = np.minimum.reduce([np.broadcast_to(ys / top, (h, w)),
                           np.broadcast_to(ys[::-1] / other, (h, w)),
                           np.broadcast_to(xs / other, (h, w)),
                           np.broadcast_to(xs[:, ::-1] / other, (h, w))])
    alpha = ((1 - np.clip(d, 0, 1)) ** 1.6 * a).astype(np.uint8)
    arr = pygame.surfarray.pixels_alpha(layer)
    arr[:, :] = alpha.T
    del arr
    surf.blit(layer, (x, y))


def _light_pool(surf, cx, cy, r, color, a=46):
    tmp = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    for i in range(10):
        rr = int(r * (1 - i / 10))
        pygame.draw.circle(tmp, (*color, int(a / 10)), (r, r), rr)
    surf.blit(tmp, (cx - r, cy - r))


def build_world_surface(font) -> pygame.Surface:
    global _world_surface
    if _world_surface is not None:
        return _world_surface
    surf = pygame.Surface((M.WORLD_W, M.WORLD_H))
    surf.fill((7, 9, 15))
    wall_d, wall_c = (20, 24, 36), (92, 102, 130)
    # 1) contours sombres + chapeau des murs
    for rects in M.CORRIDORS.values():
        for rc in rects:
            pygame.draw.rect(surf, wall_d, pygame.Rect(rc).inflate(56, 56), border_radius=16)
    for rc in M.ROOM_RECTS.values():
        pygame.draw.rect(surf, wall_d, pygame.Rect(rc).inflate(72, 72), border_radius=26)
    for rects in M.CORRIDORS.values():
        for rc in rects:
            pygame.draw.rect(surf, wall_c, pygame.Rect(rc).inflate(40, 40), border_radius=14)
    for rc in M.ROOM_RECTS.values():
        pygame.draw.rect(surf, wall_c, pygame.Rect(rc).inflate(54, 54), border_radius=22)
    # 2) murs du fond avec hauteur (faces avant), puis sol par-dessus
    for rects in M.CORRIDORS.values():
        for rc in rects:
            r = pygame.Rect(rc)
            if r.w >= r.h:
                _wall_face(surf, r.x, r.y - 44, r.w, 44)
    for rc in M.ROOM_RECTS.values():
        r = pygame.Rect(rc)
        _wall_face(surf, r.x - 26, r.y - WALL_H, r.w + 52, WALL_H)
    for rects in M.CORRIDORS.values():
        for rc in rects:
            r = pygame.Rect(rc)
            pygame.draw.rect(surf, (104, 110, 128), r)
            if r.w >= r.h:
                for x in range(r.x, r.right, 44):
                    pygame.draw.line(surf, (90, 96, 114), (x, r.y), (x, r.bottom), 3)
                _inner_shadow(surf, (r.x, r.y, r.w, r.h), top=40, other=1, a=130)
            else:
                for y in range(r.y, r.bottom, 44):
                    pygame.draw.line(surf, (90, 96, 114), (r.x, y), (r.right, y), 3)
                _inner_shadow(surf, (r.x, r.y, r.w, r.h), top=1, other=34, a=110)
    for r, rc in M.ROOM_RECTS.items():
        rect = pygame.Rect(rc)
        spec = scene.SPEC[r]
        scene._floor_pattern(surf, rect, spec["floor"][0], spec["floor"][1], spec["pat"])
        # éclairage ambiant : halo au centre, ombre sur les bords et sous le mur du fond
        cxr, cyr = rect.centerx, rect.centery
        _light_pool(surf, cxr, cyr, int(max(rect.w, rect.h) * 0.62), (255, 246, 220), 40)
        _inner_shadow(surf, rect, top=90, other=40, a=135)
        pygame.draw.rect(surf, (20, 24, 32), rect, 3, border_radius=8)
        for p in sorted(spec["props"], key=lambda q: q[2]):
            px, py = scene._pt(rect, p[1], p[2])
            _prop_scaled(surf, p[0], px, py, p[3] if len(p) > 3 else None)
        t = M.task_terminal(r)
        if t is not None:
            sx, sy = int(t[0]), int(t[1])
            scene._shadow(surf, sx, sy + 22, 36, 9, 90)
            scene._box(surf, sx - 26, sy - 6, 52, 26, 14, (96, 110, 138), (36, 42, 58))
            pygame.draw.rect(surf, (70, 210, 230), (sx - 20, sy - 1, 40, 14), border_radius=3)
            pygame.draw.rect(surf, (190, 250, 255), (sx - 20, sy - 1, 40, 5), border_radius=3)
            _light_pool(surf, sx, sy, 90, (90, 220, 240), 60)
        ts = font.render(ROOMS[r].upper(), True, (200, 212, 240))
        surf.blit(ts, (rect.x + 14, rect.y - WALL_H + 24))
    _world_surface = surf
    return surf


def mini_surface(surf, size):
    if size not in _mini_cache:
        _mini_cache[size] = pygame.transform.smoothscale(surf, size)
    return _mini_cache[size]


def make_vignette(radius):
    s = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)
    yy, xx = np.mgrid[0:2 * radius, 0:2 * radius]
    d = np.hypot(xx - radius, yy - radius) / radius
    a = np.clip((d - 0.30) / 0.70, 0, 1) ** 1.4 * 236
    arr = pygame.surfarray.pixels_alpha(s)
    arr[:, :] = a.T.astype(np.uint8)
    del arr
    return s


# ------------------------------------------------------------------------ appli
class RTApp(UIBase):
    def __init__(self, cfg: Config, crew_ctrl, imp_ctrl, rt: RTConfig | None = None, human: int | None = 0,
                 human_role: str = "crew", seed: int | None = None, god: bool = False, speed: float = 1.0,
                 crew_vote=None, imp_vote=None):
        pygame.init()
        pygame.display.set_caption("Among Us IA — temps réel")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.cfg = cfg
        self.crew_ctrl, self.imp_ctrl = crew_ctrl, imp_ctrl
        self.crew_vote, self.imp_vote = crew_vote, imp_vote      # votes appris (None -> règle heuristique)
        self.human = human
        self.human_role = human_role
        self.god = god
        self.speed = speed
        self.inject: deque = deque()
        self.fonts: dict = {}
        self.world = RTWorld(cfg, rt, seed=seed)
        self.env = self.world
        self.log: deque = deque(maxlen=12)
        self.score = {"crew": 0, "imp": 0}
        self.known_dead: set[int] = set()
        self.paused = False
        self.over = False
        self.follow: int | None = None
        self.anim_t = 0.0
        self.cam = [0.0, 0.0]
        self.show_map = False
        self.flash = None
        self.anim_scale = 1.0           # 0 = pas d'animations (tests)
        self.static = build_world_surface(self.font(22, True))
        self.vignette = make_vignette(LIGHT_R)
        self.mini = mini_surface(self.static, (300, int(300 * M.WORLD_H / M.WORLD_W)))
        self.btn_use = pygame.Rect(W - 150, H - 130, 120, 100)
        self.btn_report = pygame.Rect(W - 150, H - 244, 120, 100)
        self.new_game()

    # ------------------------------------------------------------ nouvelle partie
    def new_game(self):
        w = self.world
        imposter = None
        if self.human is not None:
            role = self.human_role
            if role == "random":
                role = w.rng.choice(["crew", "imposter"])
            imposter = self.human if role == "imposter" else w.rng.choice(
                [p for p in range(w.n) if p != self.human])
        w.vote_callbacks.clear()
        w.reset(imposter)
        w.human = self.human
        w.set_controllers(self.crew_ctrl, self.imp_ctrl, crew_vote=self.crew_vote, imp_vote=self.imp_vote)
        if self.human is not None:
            w.vote_callbacks[self.human] = self.human_vote
        self.log.clear()
        self.known_dead = set()
        self.over = False
        self.paused = False
        self.show_map = False
        self.flash = None
        self.follow = None
        self.add_log("Nouvelle partie. Tout le monde est dans la Cafétéria.", DIM)
        if self.human is not None:
            if w.imp == self.human:
                self.add_log("Vous êtes l'IMPOSTEUR : tuez sans témoin !", BAD)
            else:
                self.add_log("Vous êtes CREWMATE : faites vos tâches (E / UTILISER).", GOOD)
        fx, fy = self.focus_pos()
        self.cam = [fx - W / 2, fy - H / 2]

    # ----------------------------------------------------------------- utilitaires
    @property
    def human_alive(self):
        return self.human is not None and self.world.alive[self.human]

    @property
    def human_is_imp(self):
        return self.human is not None and self.world.imp == self.human

    def focus_player(self):
        w = self.world
        if self.human is not None:
            return self.human
        if self.follow is not None and w.alive[self.follow]:
            return self.follow
        alive = w.alive_players()
        self.follow = alive[0] if alive else 0
        return self.follow

    def focus_pos(self):
        p = self.focus_player()
        return self.world.x[p], self.world.y[p]

    def full_view(self):
        return self.god or self.human is None or not self.human_alive

    def read_input(self):
        k = pygame.key.get_pressed()
        dx = (k[pygame.K_RIGHT] or k[pygame.K_d]) - (k[pygame.K_LEFT] or k[pygame.K_q] or k[pygame.K_a])
        dy = (k[pygame.K_DOWN] or k[pygame.K_s]) - (k[pygame.K_UP] or k[pygame.K_z] or k[pygame.K_w])
        use = bool(k[pygame.K_e] or k[pygame.K_SPACE])
        if not use and pygame.mouse.get_pressed()[0] and self.btn_use.collidepoint(pygame.mouse.get_pos()):
            use = True
        return dict(dx=float(dx), dy=float(dy), use=use)

    # -------------------------------------------------------------- événements
    def process_events(self):
        w, h = self.world, self.human
        for ev in w.drain_events():
            t = ev["type"]
            if t == "task":
                if ev["player"] == h:
                    left = len(w.tasks[h])
                    self.add_log(f"Tâche terminée : {ROOMS[ev['room']]} ({left} restante(s))", GOOD)
            elif t == "kill":
                v = ev["victim"]
                if h is not None and (v == h or ev["killer"] == h or h in ev["witnesses"]):
                    self.flash = ((200, 30, 30), pygame.time.get_ticks() + 350)
                if h is not None and ev["killer"] == h:
                    self.add_log(f"Vous avez tué {PLAYER_NAMES[v]} !", BAD)
                    if ev["witnesses"]:
                        self.add_log("... mais il y avait des témoins !", WARN)
                elif v == h:
                    self.add_log(f"Vous avez été tué par {PLAYER_NAMES[ev['killer']]} !", BAD)
                    self.add_log("Vous êtes un fantôme : explorez librement.", DIM)
                elif h is not None and h in ev["witnesses"]:
                    self.add_log(f"Vous avez VU {PLAYER_NAMES[ev['killer']]} tuer {PLAYER_NAMES[v]} !", BAD)
                elif self.full_view():
                    self.add_log(f"{PLAYER_NAMES[ev['killer']]} tue {PLAYER_NAMES[v]} ({ROOMS[ev['room']]})", BAD)
            elif t == "meeting":
                k = self.anim_scale
                if k > 0:
                    self.banner("CORPS DÉCOUVERT !", f"Réunion d'urgence — {PLAYER_NAMES[ev['reporter']]} a signalé un corps",
                                int((900 if self.human is not None else 500) * k))
                self.show_meeting_result(ev)
                for v in {b[1] for b in ev["bodies"]}:
                    self.known_dead.add(v)
                if ev["ejected"] is not None:
                    self.known_dead.add(ev["ejected"])
                    if k > 0:
                        self.eject_animation(ev)
                    verdict = "ÉTAIT l'imposteur" if ev["was_imposter"] else "n'était PAS l'imposteur"
                    self.add_log(f"Réunion : {PLAYER_NAMES[ev['ejected']]} éjecté(e) — {verdict}.",
                                 GOOD if ev["was_imposter"] else WARN)
                else:
                    self.add_log("Réunion : personne n'est éjecté.", DIM)
                fx, fy = self.focus_pos()
                self.cam = [fx - W / 2, fy - H / 2]
            elif t == "end":
                self.over = True

    def eject_animation(self, ev):
        """Le robot éjecté est projeté dans l'espace."""
        e = ev["ejected"]
        rng = random.Random(e)
        stars = [(rng.randrange(W), rng.randrange(H), rng.choice((1, 1, 2, 3))) for _ in range(160)]
        total = 2.8 * self.anim_scale
        start = pygame.time.get_ticks()
        sprite = pygame.Surface((140, 160), pygame.SRCALPHA)
        scene.draw_bot(sprite, 70, 120, COLORS[e], 1)
        sprite = pygame.transform.smoothscale(sprite, (280, 320))
        txt = f"{PLAYER_NAMES[e]} a été éjecté(e)."
        sub = ("... c'ÉTAIT l'imposteur." if ev["was_imposter"] else "... ce n'était PAS l'imposteur.")
        while True:
            t = (pygame.time.get_ticks() - start) / 1000.0
            if t > total:
                return
            self.poll()
            self.screen.fill((4, 5, 12))
            for i, (sx, sy, sz) in enumerate(stars):
                x = (sx - t * 40 * sz) % W
                pygame.draw.circle(self.screen, (200, 205, 230), (int(x), sy), 1 if sz < 3 else 2)
            f = t / total
            x = -200 + f * (W + 400)
            y = H * 0.42 + math.sin(f * 5) * 40
            rot = pygame.transform.rotate(sprite, -t * 330)
            self.screen.blit(rot, rot.get_rect(center=(x, y)))
            self.text(txt, (W // 2, H - 150), 32, TXT, True, center=True)
            if f > 0.45:
                self.text(sub, (W // 2, H - 100), 24, GOOD if ev["was_imposter"] else BAD, True, center=True)
            pygame.display.flip()
            self.clock.tick(60)

    # --------------------------------------------------------------------- dessin
    def world_to_screen(self, x, y):
        return x - self.cam[0], y - self.cam[1]

    def visible(self, p, q_for_view):
        """Le joueur p est-il visible depuis le point de vue (humain vivant) ?"""
        if self.full_view():
            return True
        w = self.world
        h = self.human
        if p == h:
            return True
        d = math.hypot(w.x[p] - w.x[h], w.y[p] - w.y[h])
        return d < LIGHT_R and M.line_of_sight(w.x[h], w.y[h], w.x[p], w.y[p])

    def update_camera(self, dt, snap=False):
        fx, fy = self.focus_pos()
        tx = min(max(fx - W / 2, 0), M.WORLD_W - W)
        ty = min(max(fy - H / 2, 0), M.WORLD_H - H)
        k = 1.0 if snap else min(1.0, 9 * dt)
        self.cam[0] += (tx - self.cam[0]) * k
        self.cam[1] += (ty - self.cam[1]) * k

    def draw(self):
        w = self.world
        cx, cy = int(self.cam[0]), int(self.cam[1])
        self.screen.blit(self.static, (0, 0), area=pygame.Rect(cx, cy, W, H))
        h = self.human
        # bornes de tâche du joueur
        if self.human_alive and not self.human_is_imp:
            pulse = 1.0 + 0.15 * math.sin(self.anim_t * 5)
            for r in w.tasks[h]:
                tx, ty = M.task_terminal(r)
                sx, sy = tx - cx, ty - cy
                if -80 < sx < W + 80 and -80 < sy < H + 80:
                    pygame.draw.circle(self.screen, WARN, (int(sx), int(sy)), int(44 * pulse), 3)
                    pygame.draw.rect(self.screen, WARN, (sx - 11, sy - 66, 22, 24), border_radius=5)
                    self.text("!", (sx, sy - 54), 18, (30, 30, 30), True, center=True)
        # cadavres
        for (_room, v) in w.bodies:
            bx, by = w.body_pos.get(v, (None, None))
            if bx is None:
                continue
            if self.full_view() or (math.hypot(bx - w.x[h], by - w.y[h]) < LIGHT_R and M.line_of_sight(w.x[h], w.y[h], bx, by)):
                scene.draw_bot(self.screen, bx - cx, by - cy + 16, COLORS[v], 1, dead=True,
                               label=PLAYER_NAMES[v] + " (mort)", font=self.font(12, True))
        # personnages
        order = sorted((p for p in range(w.n) if w.alive[p]), key=lambda q: w.y[q])
        for p in order:
            if not self.visible(p, h):
                continue
            sx, sy = w.x[p] - cx, w.y[p] - cy
            if -80 < sx < W + 80 and -100 < sy < H + 100:
                scene.draw_bot(self.screen, sx, sy + 16, COLORS[p], w.face[p], moving=w.moving[p], t=self.anim_t,
                               working=w.working[p], label=PLAYER_NAMES[p], font=self.font(12, True), you=(p == h))
                if self.god and p == w.imp:
                    pygame.draw.circle(self.screen, BAD, (int(sx), int(sy) - 62), 4)
        if h is not None and not w.alive[h]:      # fantôme : translucide
            tmp = pygame.Surface((120, 140), pygame.SRCALPHA)
            scene.draw_bot(tmp, 60, 100, COLORS[h], w.face[h], moving=w.moving[h], t=self.anim_t,
                           label="Vous (fantôme)", font=self.font(12, True))
            tmp.set_alpha(130)
            self.screen.blit(tmp, (w.x[h] - cx - 60, w.y[h] - cy - 84))
        # obscurité + lumière (vision limitée)
        if not self.full_view():
            fx, fy = w.x[h], w.y[h]
            dark = pygame.Surface((W, H), pygame.SRCALPHA)
            dark.fill((3, 5, 11, 236))
            poly = M.visibility_polygon(fx, fy, LIGHT_R)
            pts = [(float(px - cx), float(py - cy)) for px, py in poly]
            pygame.draw.polygon(dark, (3, 5, 11, 0), pts)
            dark.blit(self.vignette, (int(fx - cx) - LIGHT_R, int(fy - cy) - LIGHT_R), special_flags=pygame.BLEND_RGBA_MAX)
            small = pygame.transform.smoothscale(dark, (W // 5, H // 5))      # bord d'ombre adouci
            self.screen.blit(pygame.transform.smoothscale(small, (W, H)), (0, 0))
        self.draw_hud()
        if self.show_map:
            self.draw_big_map()
        if self.flash and pygame.time.get_ticks() < self.flash[1]:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((*self.flash[0], 80))
            self.screen.blit(ov, (0, 0))
        pygame.display.flip()

    # ------------------------------------------------------------------------ HUD
    def panel(self, rect, alpha=170):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (14, 18, 30, alpha), s.get_rect(), border_radius=12)
        pygame.draw.rect(s, (90, 100, 140, 200), s.get_rect(), 2, border_radius=12)
        self.screen.blit(s, rect.topleft)

    def draw_hud(self):
        w, h = self.world, self.human
        # barre de tâches globale
        bar = pygame.Rect(20, 18, 330, 30)
        self.panel(bar.inflate(10, 10))
        frac = w.tasks_fraction()
        pygame.draw.rect(self.screen, (40, 48, 70), bar, border_radius=8)
        if frac > 0:
            pygame.draw.rect(self.screen, GOOD, (bar.x, bar.y, int(bar.w * frac), bar.h), border_radius=8)
        self.text("TÂCHES TERMINÉES", bar.center, 15, (14, 20, 16) if frac > 0.5 else TXT, True, center=True)
        y = 64
        if self.human_alive and not self.human_is_imp:
            rows = sorted(w.tasks[h])
            self.panel(pygame.Rect(20, y - 6, 250, 30 + 22 * max(1, len(rows))), 150)
            self.text("Mes tâches", (32, y), 15, TXT, True)
            for r in rows:
                y += 22
                prog = w.work[h].get(r, 0.0) / w.rt.task_time
                self.text(f"• {ROOMS[r]}" + (f"  {int(prog * 100)}%" if prog > 0 else ""), (36, y + 6), 14, WARN)
            if not rows:
                self.text("Tout est fait !", (36, y + 28), 14, GOOD)
        elif self.human_alive:
            self.panel(pygame.Rect(20, y - 6, 300, 74), 150)
            self.text("IMPOSTEUR", (32, y), 16, BAD, True)
            self.text(f"Crewmates en vie : {len(w.crew_alive())}  (victoire <= {self.cfg.crew_left_to_lose})",
                      (32, y + 26), 13, TXT)
            self.text("Tuez à l'écart des témoins !", (32, y + 46), 12, DIM)
        # chrono + rôle
        remaining = max(0, int(w.rt.max_time - w.time))
        self.text(f"{remaining // 60}:{remaining % 60:02d}", (W - 24, 14), 24, TXT, True, right=True)
        label = ("SPECTATEUR — suivi : " + PLAYER_NAMES[self.focus_player()] + "  (Tab)") if self.human is None else (
            "FANTÔME — vous voyez tout" if not self.human_alive else "")
        if label:
            self.text(label, (W // 2, 18), 16, DIM, True, center=True)
        # minicarte
        mm = self.mini.get_rect(topright=(W - 20, 52))
        self.panel(mm.inflate(10, 10), 190)
        self.screen.blit(self.mini, mm)
        sc = mm.w / M.WORLD_W
        if self.human_alive and not self.human_is_imp:
            for r in w.tasks[h]:
                tx, ty = M.task_terminal(r)
                pygame.draw.rect(self.screen, WARN, (mm.x + tx * sc - 3, mm.y + ty * sc - 3, 6, 6))
        for p in range(w.n):
            if not w.alive[p] or not (self.full_view() or p == h):
                continue
            pygame.draw.circle(self.screen, COLORS[p], (int(mm.x + w.x[p] * sc), int(mm.y + w.y[p] * sc)), 4)
            if p == h:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(mm.x + w.x[p] * sc), int(mm.y + w.y[p] * sc)), 6, 1)
        # journal
        yy = H - 24 - 18 * min(5, len(self.log))
        for msg, col in list(self.log)[-5:]:
            ts = self.font(13).render(msg, True, col)
            bg = ts.get_rect(topleft=(20, yy)).inflate(10, 4)
            s = pygame.Surface(bg.size, pygame.SRCALPHA)
            s.fill((10, 12, 20, 150))
            self.screen.blit(s, bg.topleft)
            self.screen.blit(ts, (25, yy + 2))
            yy += 18
        # boutons
        if self.human_alive:
            self.draw_buttons()
        # aide
        self.text("ZQSD/flèches : bouger   E/Espace : utiliser   K : kill   R : signaler   M : carte   V : debug   Échap",
                  (W // 2, H - 14), 11, DIM, center=True)

    def draw_buttons(self):
        w, h = self.world, self.human
        mouse = pygame.mouse.get_pos()
        if self.human_is_imp:
            r = self.btn_use
            ready = w.human_can_kill()
            cd = int(math.ceil(w.cd_s))
            col = (205, 50, 50) if ready else (84, 70, 76)
            pygame.draw.rect(self.screen, (20, 14, 16), r.move(0, 5), border_radius=18)
            pygame.draw.rect(self.screen, col, r, border_radius=18)
            pygame.draw.rect(self.screen, (255, 200, 200) if ready else (120, 110, 116), r, 3, border_radius=18)
            if w.cd_s > 0:
                self.text(str(cd), r.center, 40, (255, 240, 240), True, center=True)
                self.text("recharge", (r.centerx, r.bottom - 16), 12, (255, 220, 220), center=True)
            else:
                self.text("KILL", (r.centerx, r.centery - 6), 28, (255, 240, 240), True, center=True)
                self.text("[K]" if ready else "trop loin", (r.centerx, r.bottom - 18), 12, (255, 220, 220), center=True)
            return
        near = w.my_task_nearby(h)
        r = self.btn_use
        col = (236, 196, 62) if near is not None else (74, 78, 92)
        if near is not None and r.collidepoint(mouse):
            col = (255, 220, 100)
        pygame.draw.rect(self.screen, (16, 18, 26), r.move(0, 5), border_radius=18)
        pygame.draw.rect(self.screen, col, r, border_radius=18)
        pygame.draw.rect(self.screen, (255, 244, 200) if near is not None else (110, 114, 130), r, 3, border_radius=18)
        self.text("UTILISER", (r.centerx, r.centery - 8), 20, (30, 28, 20) if near is not None else (150, 154, 170),
                  True, center=True)
        self.text("[E] maintenir" if near is not None else "approchez-vous", (r.centerx, r.bottom - 20), 12,
                  (60, 50, 20) if near is not None else (130, 134, 150), center=True)
        if near is not None:
            prog = w.work[h].get(near, 0.0) / w.rt.task_time
            if prog > 0:
                pygame.draw.rect(self.screen, (40, 40, 50), (r.x + 8, r.y - 14, r.w - 16, 8), border_radius=4)
                pygame.draw.rect(self.screen, GOOD, (r.x + 8, r.y - 14, int((r.w - 16) * prog), 8), border_radius=4)
        rr = self.btn_report
        can = w.human_can_report()
        col = (226, 84, 70) if can else (74, 78, 92)
        if can and rr.collidepoint(mouse):
            col = (246, 120, 100)
        pygame.draw.rect(self.screen, (16, 18, 26), rr.move(0, 5), border_radius=18)
        pygame.draw.rect(self.screen, col, rr, border_radius=18)
        pygame.draw.rect(self.screen, (255, 210, 200) if can else (110, 114, 130), rr, 3, border_radius=18)
        self.text("SIGNALER", (rr.centerx, rr.centery - 8), 19, (255, 240, 236) if can else (150, 154, 170), True, center=True)
        self.text("[R]" if can else "aucun corps", (rr.centerx, rr.bottom - 20), 12, (255, 226, 220) if can else (130, 134, 150),
                  center=True)

    def draw_big_map(self):
        w, h = self.world, self.human
        box = pygame.Rect(40, 40, W - 80, H - 80)
        self.panel(box, 235)
        size = (box.w - 30, int((box.w - 30) * M.WORLD_H / M.WORLD_W))
        surf = mini_surface(self.static, size)
        pos = surf.get_rect(center=box.center)
        self.screen.blit(surf, pos)
        sc = size[0] / M.WORLD_W
        if self.human_alive and not self.human_is_imp:
            for r in w.tasks[h]:
                tx, ty = M.task_terminal(r)
                pygame.draw.rect(self.screen, WARN, (pos.x + tx * sc - 6, pos.y + ty * sc - 6, 12, 12))
        for p in range(w.n):
            if not w.alive[p] or not (self.full_view() or p == h):
                continue
            c = (int(pos.x + w.x[p] * sc), int(pos.y + w.y[p] * sc))
            pygame.draw.circle(self.screen, COLORS[p], c, 8)
            if p == h:
                pygame.draw.circle(self.screen, (255, 255, 255), c, 11, 2)
        self.text("CARTE  (M pour fermer)", (box.centerx, box.y + 12), 16, TXT, True, center=True)

    # ------------------------------------------------------------ boucle principale
    def run(self):
        w = self.world
        acc = 0.0
        while True:
            self.over = False
            while not self.over:
                dt = min(self.clock.tick(60) / 1000.0, 0.1)
                self.anim_t += dt
                for e in self.poll():
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        if e.key == pygame.K_m:
                            self.show_map = not self.show_map
                        elif e.key == pygame.K_v:
                            self.god = not self.god
                        elif e.key == pygame.K_r and self.human_alive:
                            w.human_report()
                        elif e.key in (pygame.K_k, pygame.K_f) and self.human_alive:
                            w.human_kill()
                        elif e.key == pygame.K_TAB and self.human is None:
                            alive = w.alive_players()
                            if alive:
                                cur = self.follow if self.follow in alive else alive[0]
                                self.follow = alive[(alive.index(cur) + 1) % len(alive)]
                        elif e.key == pygame.K_p and self.human is None:
                            self.paused = not self.paused
                        elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                            self.speed = min(12.0, self.speed * 1.5)
                        elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.speed = max(0.25, self.speed / 1.5)
                    elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.human_alive:
                        if self.btn_report.collidepoint(e.pos):
                            w.human_report()
                        elif self.human_is_imp and self.btn_use.collidepoint(e.pos):
                            w.human_kill()
                self.process_events()
                if self.over:
                    break
                if self.human is not None:
                    w.human_input = self.read_input()
                if not self.paused:
                    acc += dt * (self.speed if self.human is None else 1.0)
                    n = 0
                    while acc >= TICK and n < 12 and w.winner is None:
                        w.update(TICK)
                        self.process_events()
                        acc -= TICK
                        n += 1
                        if self.over:
                            break
                    if n >= 12:
                        acc = 0.0
                if w.winner is not None:
                    self.over = True
                self.update_camera(dt)
                self.draw()
            self.process_events()
            if not self.end_screen():
                pygame.quit()
                return
            self.new_game()
            acc = 0.0
