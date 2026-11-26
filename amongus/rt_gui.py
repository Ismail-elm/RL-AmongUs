"""Interface TEMPS RÉEL : déplacement libre (ZQSD / WASD / flèches), caméra, vision limitée, boutons."""
from __future__ import annotations

import math
import random
import os
import sys
from collections import deque

import numpy as np
import pygame

from . import rt_map as M
from . import canvas as D
from . import scene
from . import style as ST
from .config import Config, PLAYER_NAMES
from .ui import (UIBase, BAD, BG, COLORS, DIM, GOOD, PANEL, TXT, WARN, H, W, meeting_background)
from .mapdata import ROOMS
from .rt_world import RTConfig, RTWorld

LIGHT_R = 470


def enable_dpi_awareness():
    """Windows avec mise à l'échelle (125 %, 150 %...) : sans cela, le jeu croit l'écran plus petit qu'il n'est
    (1536 x 864 au lieu de 1920 x 1080), dessine en basse résolution et Windows l'agrandit (flou)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)       # par écran
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _new_window(pos, size, flags=0):
    """Ouvre une fenêtre neuve à la position voulue (SDL ne lit SDL_VIDEO_WINDOW_POS qu'à la création)."""
    if pygame.display.get_surface() is not None:
        pygame.display.quit()
        pygame.display.init()
        pygame.display.set_caption("Among Us IA — temps réel")
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{pos[0]},{pos[1]}"
    return pygame.display.set_mode(size, flags)
TICK = 1 / 30

_world_surface = None
_mini_cache: dict = {}


# ------------------------------------------------------------------ rendu statique
def _prop_scaled(surf, kind, cx, cy, opt, s=0.7):
    """Accessoire de décor, réduit, avec un contour noir (style cartoon)."""
    if isinstance(surf, D.Canvas):                  # tracé directement à la bonne taille : net
        tmp = D.sprite(360, 360, s, lambda c: scene._prop(c, kind, 180, 180, opt), surf.k)
        tmp = ST.outline_sprite(tmp, max(2, round(3 * surf.k)))
        surf.blit_native(tmp, center=(cx, cy))
        return
    tmp = pygame.Surface((360, 360), pygame.SRCALPHA)
    scene._prop(tmp, kind, 180, 180, opt)
    tmp = pygame.transform.smoothscale(tmp, (int(360 * s), int(360 * s)))
    tmp = ST.outline_sprite(tmp, 3)
    surf.blit(tmp, tmp.get_rect(center=(cx, cy)))


WALL_H = M.WALL_H


def _wall_face(surf, x, y, w, h):
    """Mur du fond vu de face : aplat, joints verticaux, liseré clair en haut, trait noir au pied."""
    D.rect(surf, ST.WALL, (x, y, w, h))
    D.rect(surf, ST.shade(ST.WALL, 1.25), (x, y, w, 6))
    for sx in range(x + 48, x + w - 10, 96):
        D.line(surf, ST.WALL_DARK, (sx, y + 8), (sx, y + h - 2), 3)
    D.line(surf, ST.OUTLINE, (x, y + h - 2), (x + w, y + h - 2), 4)


def _floor(surf, rect, tint):
    """Sol métallique en dalles, légèrement teinté selon la salle."""
    base = ST.mix(ST.STEEL, tint, 0.28)
    line = ST.mix(ST.STEEL_LINE, ST.shade(tint, 0.8), 0.28)
    D.rect(surf, base, rect)
    for x in range(rect.x + 60, rect.right, 60):
        D.line(surf, line, (x, rect.y), (x, rect.bottom), 2)
    for y in range(rect.y + 60, rect.bottom, 60):
        D.line(surf, line, (rect.x, y), (rect.right, y), 2)
    for x in range(rect.x + 30, rect.right, 120):          # petits rivets
        for y in range(rect.y + 30, rect.bottom, 120):
            D.circle(surf, line, (x, y), 3)


def build_world_surface(font) -> pygame.Surface:
    """Décor du vaisseau à l'échelle 1 (sert à la minicarte et de repère logique)."""
    global _world_surface
    if _world_surface is None:
        _world_surface = pygame.Surface((M.WORLD_W, M.WORLD_H))
        draw_world(_world_surface, font)
    return _world_surface


def draw_world(surf, font):
    """Trace le décor sur `surf` (surface ordinaire ou Canvas à l'échelle de l'écran)."""
    surf.fill(ST.SPACE)
    rng = random.Random(5)
    for _ in range(900):                                     # étoiles
        c = rng.randint(90, 200)
        D.circle(surf, (c, c, min(255, c + 25)), (rng.randrange(M.WORLD_W), rng.randrange(M.WORLD_H)),
                           rng.choice((1, 1, 1, 2)))
    rooms = [pygame.Rect(rc) for rc in M.ROOM_RECTS.values()]
    corrs = [pygame.Rect(rc) for rects in M.CORRIDORS.values() for rc in rects]
    # 1) contour noir épais, puis épaisseur des murs
    for r in corrs:
        D.rect(surf, ST.OUTLINE, r.inflate(48, 48), border_radius=14)
    for r in rooms:
        D.rect(surf, ST.OUTLINE, r.inflate(64, 64), border_radius=20)
    for r in corrs:
        D.rect(surf, ST.WALL_DARK, r.inflate(36, 36), border_radius=10)
    for r in rooms:
        D.rect(surf, ST.WALL_DARK, r.inflate(50, 50), border_radius=16)
    # 2) murs du fond (faces avant), puis les sols par-dessus
    for r in corrs:
        if r.w >= r.h:
            _wall_face(surf, r.x, r.y - 44, r.w, 44)
    for r in rooms:
        _wall_face(surf, r.x - 25, r.y - WALL_H, r.w + 50, WALL_H)
    for r in corrs:
        D.rect(surf, ST.CORRIDOR, r)
        if r.w >= r.h:
            for x in range(r.x, r.right, 44):
                D.line(surf, ST.STEEL_LINE, (x, r.y), (x, r.bottom), 3)
            D.rect(surf, ST.shade(ST.CORRIDOR, 0.82), (r.x, r.y, r.w, 10))     # ombre du mur
        else:
            for y in range(r.y, r.bottom, 44):
                D.line(surf, ST.STEEL_LINE, (r.x, y), (r.right, y), 3)
    for rid, rc in M.ROOM_RECTS.items():
        rect = pygame.Rect(rc)
        spec = scene.SPEC[rid]
        _floor(surf, rect, spec["floor"][0])
        D.rect(surf, ST.shade(ST.mix(ST.STEEL, spec["floor"][0], 0.28), 0.8), (rect.x, rect.y, rect.w, 14))
        for p in sorted(spec["props"], key=lambda q: q[2]):
            px, py = scene._pt(rect, p[1], p[2])
            _prop_scaled(surf, p[0], px, py, p[3] if len(p) > 3 else None)
        t = M.task_terminal(rid)
        if t is not None:                                      # borne de tâche
            sx, sy = int(t[0]), int(t[1])
            ST.orect(surf, (74, 82, 102), (sx - 26, sy - 10, 52, 30), 5, 3)
            ST.orect(surf, (110, 228, 245), (sx - 19, sy - 4, 38, 13), 3, 2)
            D.rect(surf, (230, 255, 255), (sx - 16, sy - 2, 12, 3))
        D.label(surf, font, ROOMS[rid].upper(), (255, 255, 255), 3, topleft=(rect.x + 12, rect.y - WALL_H + 18))


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
                 crew_vote=None, imp_vote=None, imp_claim=None, fullscreen=False, menu=False):
        enable_dpi_awareness()
        pygame.init()
        pygame.display.set_caption("Among Us IA — temps réel")
        self.fullscreen = fullscreen
        self.screen = None
        self.menu = menu
        self.free_cam = human is None           # spectateur : caméra libre au départ
        self._mm_rect = None
        self._bigmap = None
        self.clock = pygame.time.Clock()
        self.cfg = cfg
        self.crew_ctrl, self.imp_ctrl = crew_ctrl, imp_ctrl
        self.crew_vote, self.imp_vote = crew_vote, imp_vote      # votes appris (None -> règle heuristique)
        self.imp_claim = imp_claim                                # déclaration apprise de l'imposteur
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
        self._vignettes = {LIGHT_R: self.vignette}
        self.mini = mini_surface(self.static, (240, int(240 * M.WORLD_H / M.WORLD_W)))
        self._hi_world = {}
        self.setup_display()
        self.btn_use = pygame.Rect(W - 150, H - 130, 120, 100)
        self.btn_report = pygame.Rect(W - 150, H - 244, 120, 100)
        self.btn_emergency = pygame.Rect(W - 290, H - 130, 120, 100)
        self.btn_vent = pygame.Rect(W - 290, H - 130, 120, 100)
        self.btn_sabotage = pygame.Rect(W - 290, H - 244, 120, 100)
        self.btn_map = pygame.Rect(W - 140, 206, 120, 34)
        self.new_game()

    # ------------------------------------------------------------ écran
    def setup_display(self, fullscreen=None):
        """Ouvre l'écran à sa VRAIE résolution et y trace tout nativement (textes et formes nets)."""
        if fullscreen is not None:
            self.fullscreen = fullscreen
        sizes = pygame.display.get_desktop_sizes() if hasattr(pygame.display, "get_desktop_sizes") else []
        dw, dh = sizes[0] if sizes else (W, H)
        if self.fullscreen:
            # plein écran « fenêtre sans bordure » à la taille du bureau : la résolution de l'écran ne change pas,
            # donc les autres applications ne sont pas redimensionnées (contrairement au plein écran exclusif)
            real = _new_window((0, 0), (dw, dh), pygame.NOFRAME)
        else:
            f = max(1.0, min(dw * 0.92 / W, dh * 0.85 / H))
            size = (round(W * f), round(H * f))
            real = _new_window(((dw - size[0]) // 2, max(0, (dh - size[1]) // 3)), size)
        rw, rh = real.get_size()
        k = min(rw / W, rh / H)
        self.screen = D.Canvas(real, k, ((rw - W * k) / 2, (rh - H * k) / 2), (W, H))
        # décor du vaisseau tracé à la résolution de l'écran (plafonné pour la mémoire, au-delà on agrandit)
        hk = min(k, 1.6)
        hi = self._hi_world.get(hk)
        if hi is None:
            hi = pygame.Surface((round(M.WORLD_W * hk), round(M.WORLD_H * hk)))
            draw_world(D.Canvas(hi, hk, (0, 0), (M.WORLD_W, M.WORLD_H)), self.font(22, True))
            self._hi_world = {hk: hi}
        self.screen.register(self.static, hi, hk)
        self._register_map(self.mini)

    def _register_map(self, surf):
        """Version nette d'une carte réduite (minicarte, grande carte)."""
        hk = self.screen.k
        size = (max(1, round(surf.get_width() * hk)), max(1, round(surf.get_height() * hk)))
        hi = pygame.transform.smoothscale(next(iter(self._hi_world.values())), size)
        self.screen.register(surf, hi, hk)

    def toggle_fullscreen(self):
        self.setup_display(not self.fullscreen)

    # ------------------------------------------------------------ nouvelle partie
    def layout_buttons(self):
        """Boutons d'action en bas à droite, comme dans le jeu : la colonne de droite (utiliser / tuer, signaler)
        est commune ; l'imposteur a en plus conduit et sabotage, et le bouton d'urgence se décale."""
        imp = self.human is not None and self.human in self.world.imps
        self.btn_emergency = pygame.Rect(W - (430 if imp else 290), H - 130, 120, 100)

    def new_game(self):
        w = self.world
        imposter = None
        if self.human is not None:
            role = self.human_role
            if role == "random":
                role = w.rng.choice(["crew", "imposter"])
            others = [p for p in range(w.n) if p != self.human]
            if role == "imposter":       # l'humain + des partenaires IA
                imposter = [self.human] + w.rng.sample(others, w.n_imp - 1)
            else:                        # que des imposteurs IA
                imposter = w.rng.sample(others, w.n_imp)
        w.vote_callbacks.clear()
        w.claim_callbacks.clear()
        w.reset(imposter)
        w.human = self.human
        w.set_controllers(self.crew_ctrl, self.imp_ctrl, crew_vote=self.crew_vote, imp_vote=self.imp_vote,
                          imp_claim=self.imp_claim)
        if self.human is not None:
            w.vote_callbacks[self.human] = self.human_vote
            w.claim_callbacks[self.human] = self.human_claim
        self.layout_buttons()
        self.log.clear()
        self.known_dead = set()
        self.over = False
        self.paused = False
        self.show_map = False
        self.flash = None
        self.follow = None
        self.add_log("Nouvelle partie. Tout le monde est dans la Cafétéria.", DIM)
        if self.human is not None:
            if self.human in w.imps:
                mates = [PLAYER_NAMES[q] for q in w.imps if q != self.human]
                self.add_log("Vous êtes IMPOSTEUR : tuez sans témoin !"
                             + (f"  Partenaire : {', '.join(mates)}" if mates else ""), BAD)
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
        return self.human is not None and self.human in self.world.imps

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
        if not use and pygame.mouse.get_pressed()[0] and self.btn_use.collidepoint(self.mouse_pos()):
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
                self.show_meeting_result(ev)                # bannière comprise (si pas déjà montrée au vote)
                for v in {b[1] for b in ev["bodies"]}:
                    self.known_dead.add(v)
                if ev["ejected"] is not None:
                    self.known_dead.add(ev["ejected"])
                    if k > 0:
                        self.eject_animation(ev)
                    verdict = "ÉTAIT un imposteur" if ev["was_imposter"] else "n'était PAS un imposteur"
                    self.add_log(f"Réunion : {PLAYER_NAMES[ev['ejected']]} éjecté(e) — {verdict}.",
                                 GOOD if ev["was_imposter"] else WARN)
                else:
                    self.add_log("Réunion : personne n'est éjecté.", DIM)
                if not (self.human is None and self.free_cam):
                    fx, fy = self.focus_pos()
                    self.cam = [fx - W / 2, fy - H / 2]
            elif t == "lights":
                if not ev["on"]:
                    self.add_log("Les lumières sont coupées ! Réparez-les en Électricité.", WARN)
                else:
                    who = ev.get("fixer")
                    self.add_log("Lumières réparées" + (f" par {PLAYER_NAMES[who]}." if who is not None else "."), DIM)
            elif t == "vent":
                if ev["player"] == h:
                    self.add_log(f"Conduit : de {ROOMS[ev['src']]} à {ROOMS[ev['dst']]}"
                                 + (" — on vous a vu !" if ev["seen"] else ""), WARN if ev["seen"] else DIM)
                elif h is not None and h in ev["seen"]:
                    self.add_log(f"Vous avez vu {PLAYER_NAMES[ev['player']]} entrer dans un conduit !", BAD)
            elif t == "vent_seen":
                if h is not None and h in ev["seen"] and ev["player"] != h:
                    self.add_log(f"Vous avez vu {PLAYER_NAMES[ev['player']]} sortir d'un conduit !", BAD)
            elif t == "end":
                self.over = True

    def eject_animation(self, ev):
        """Le robot éjecté est projeté dans l'espace."""
        e = ev["ejected"]
        rng = random.Random(e)
        stars = [(rng.randrange(W), rng.randrange(H), rng.choice((1, 1, 2, 3))) for _ in range(160)]
        total = 2.8 * self.anim_scale
        start = pygame.time.get_ticks()
        sprite = D.sprite(140, 160, 2.0, lambda c: scene.draw_bot(c, 70, 120, COLORS[e], 1), self.screen.k)
        txt = f"{PLAYER_NAMES[e]} a été éjecté(e)."
        sub = ("... c'ÉTAIT un imposteur." if ev["was_imposter"] else "... ce n'était PAS un imposteur.")
        while True:
            t = (pygame.time.get_ticks() - start) / 1000.0
            if t > total:
                return
            self.poll()
            self.screen.fill(ST.BLUE_BG)
            self.screen.blit(meeting_background(self.screen), (0, 0))
            for i, (sx, sy, sz) in enumerate(stars):
                x = (sx - t * 40 * sz) % W
                D.circle(self.screen, (200, 205, 230), (int(x), sy), 1 if sz < 3 else 2)
            f = t / total
            x = -200 + f * (W + 400)
            y = H * 0.42 + math.sin(f * 5) * 40
            rot = pygame.transform.rotate(sprite, -t * 330)
            self.screen.blit_native(rot, center=(x, y))
            self.otext(txt, (W // 2, H - 150), 34, center=True, width=3)
            if f > 0.45:
                self.otext(sub, (W // 2, H - 100), 24, ST.GREEN if ev["was_imposter"] else ST.RED, center=True, width=3)
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
        if p in w.in_vent:
            return False
        d = math.hypot(w.x[p] - w.x[h], w.y[p] - w.y[h])
        return d < self.light_r() and M.line_of_sight(w.x[h], w.y[h], w.x[p], w.y[p])

    def light_r(self):
        """Rayon de lumière du joueur humain (plus petit pour un crewmate quand les lumières sont coupées)."""
        w, h = self.world, self.human
        if h is None or not w.alive[h]:
            return LIGHT_R
        return int(min(LIGHT_R, w.vision(h)))

    def draw_vents_and_panel(self, cx, cy):
        w = self.world
        for r in M.VENT_LINKS:
            vx, vy = M.vent_pos(r)
            sx, sy = vx - cx, vy - cy
            if -60 < sx < W + 60 and -60 < sy < H + 60:
                g = pygame.Rect(int(sx) - 26, int(sy) - 14, 52, 28)
                ST.orect(self.screen, (74, 82, 102), g, 4, 3)
                for i in range(1, 5):
                    D.line(self.screen, ST.OUTLINE, (g.x + i * 10, g.y + 4), (g.x + i * 10, g.bottom - 4), 3)
        if w.lights_off:
            fx, fy = M.fix_pos()
            sx, sy = fx - cx, fy - cy
            pulse = 1.0 + 0.2 * math.sin(self.anim_t * 6)
            D.circle(self.screen, ST.OUTLINE, (int(sx), int(sy)), int(38 * pulse), 7)
            D.circle(self.screen, ST.YELLOW, (int(sx), int(sy)), int(38 * pulse) - 2, 3)
            ST.orect(self.screen, (74, 82, 102), (sx - 16, sy - 30, 32, 40), 4, 3)
            ST.orect(self.screen, ST.YELLOW, (sx - 9, sy - 22, 18, 10), 2, 2)

    def clamp_camera(self):
        self.cam[0] = min(max(self.cam[0], -W / 6), M.WORLD_W - W * 5 / 6)
        self.cam[1] = min(max(self.cam[1], -H / 6), M.WORLD_H - H * 5 / 6)

    def center_camera(self, x, y):
        self.cam = [x - W / 2, y - H / 2]
        self.clamp_camera()

    def pan_camera(self, dt):
        """Spectateur : ZQSD / flèches déplacent la caméra (Maj : plus vite)."""
        k = pygame.key.get_pressed()
        dx = (k[pygame.K_RIGHT] or k[pygame.K_d]) - (k[pygame.K_LEFT] or k[pygame.K_q] or k[pygame.K_a])
        dy = (k[pygame.K_DOWN] or k[pygame.K_s]) - (k[pygame.K_UP] or k[pygame.K_z] or k[pygame.K_w])
        if dx or dy:
            v = 1800 if (k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]) else 800
            self.free_cam = True
            self.cam[0] += dx * v * dt
            self.cam[1] += dy * v * dt
            self.clamp_camera()

    def update_camera(self, dt, snap=False):
        if self.human is None:
            self.pan_camera(dt)                 # une flèche détache la caméra du joueur suivi
            if self.free_cam:
                return
        fx, fy = self.focus_pos()
        tx = min(max(fx - W / 2, 0), M.WORLD_W - W)
        ty = min(max(fy - H / 2, 0), M.WORLD_H - H)
        k = 1.0 if snap else min(1.0, 9 * dt)
        self.cam[0] += (tx - self.cam[0]) * k
        self.cam[1] += (ty - self.cam[1]) * k

    def draw(self, flip=True):
        w = self.world
        cx, cy = int(self.cam[0]), int(self.cam[1])
        self.screen.fill(ST.SPACE)                     # hors de la carte : l'espace (sinon restes de l'image précédente)
        self.screen.blit(self.static, (max(0, -cx), max(0, -cy)), area=pygame.Rect(max(0, cx), max(0, cy), W, H))
        h = self.human
        # bornes de tâche du joueur
        if self.human_alive and not self.human_is_imp:
            pulse = 1.0 + 0.15 * math.sin(self.anim_t * 5)
            for r in w.tasks[h]:
                tx, ty = M.task_terminal(r)
                sx, sy = tx - cx, ty - cy
                if -80 < sx < W + 80 and -80 < sy < H + 80:
                    D.circle(self.screen, WARN, (int(sx), int(sy)), int(44 * pulse), 3)
                    D.rect(self.screen, WARN, (sx - 11, sy - 66, 22, 24), border_radius=5)
                    self.text("!", (sx, sy - 54), 18, (30, 30, 30), True, center=True)
        self.draw_vents_and_panel(cx, cy)
        # cadavres
        for (_room, v) in w.bodies:
            bx, by = w.body_pos.get(v, (None, None))
            if bx is None:
                continue
            if self.full_view() or (math.hypot(bx - w.x[h], by - w.y[h]) < self.light_r() and M.line_of_sight(w.x[h], w.y[h], bx, by)):
                ghost_here = h is not None and not w.alive[h] and math.hypot(bx - w.x[h], by - w.y[h]) < 90
                scene.draw_bot(self.screen, bx - cx, by - cy + 16, COLORS[v], 1, dead=True,
                               label=None if ghost_here else PLAYER_NAMES[v] + " (mort)", font=self.font(12, True))
        # personnages
        order = sorted((p for p in range(w.n) if w.alive[p] and not (p in w.in_vent and not self.god)),
                       key=lambda q: w.y[q])
        labels = []
        for p in order:
            if not self.visible(p, h):
                continue
            sx, sy = w.x[p] - cx, w.y[p] - cy
            if -80 < sx < W + 80 and -100 < sy < H + 100:
                scene.draw_bot(self.screen, sx, sy + 16, COLORS[p], w.face[p], moving=w.moving[p], t=self.anim_t,
                               working=w.working[p])
                labels.append((p, sx, sy + 16 - 58))
        self.draw_name_labels(labels, h)
        if h is not None and not w.alive[h]:      # fantôme : translucide
            tmp = D.sprite(120, 140, 1.0, lambda c: scene.draw_bot(
                c, 60, 100, COLORS[h], w.face[h], moving=w.moving[h], t=self.anim_t,
                label="Vous (fantôme)", font=self.font(12, True)), self.screen.k)
            tmp.set_alpha(130)
            self.screen.blit_native(tmp, topleft=(w.x[h] - cx - 60, w.y[h] - cy - 84))
        # obscurité + lumière (vision limitée)
        if not self.full_view():
            fx, fy = w.x[h], w.y[h]
            dark = pygame.Surface((W, H), pygame.SRCALPHA)
            dark.fill((3, 5, 11, 236))
            lr = self.light_r()
            if lr not in self._vignettes:
                self._vignettes[lr] = make_vignette(lr)
            poly = M.visibility_polygon(fx, fy, lr)
            pts = [(float(px - cx), float(py - cy)) for px, py in poly]
            D.polygon(dark, (3, 5, 11, 0), pts)
            dark.blit(self._vignettes[lr], (int(fx - cx) - lr, int(fy - cy) - lr), special_flags=pygame.BLEND_RGBA_MAX)
            small = pygame.transform.smoothscale(dark, (W // 5, H // 5))      # bord d'ombre adouci
            self.screen.blit_stretch(small, (0, 0, W, H))
        self.draw_hud()
        if self.show_map:
            self.draw_big_map()
        if self.flash and pygame.time.get_ticks() < self.flash[1]:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((*self.flash[0], 80))
            self.screen.blit(ov, (0, 0))
        if flip:
            pygame.display.flip()

    def draw_name_labels(self, labels, h):
        """Noms au-dessus des têtes, dessinés après tous les robots ; deux noms qui se chevauchent sont décalés vers le
        haut au lieu de se superposer."""
        w, f = self.world, self.font(14, True)
        placed = []
        for p, x, y in sorted(labels, key=lambda t: -t[2]):          # du plus bas au plus haut à l'écran
            tw, th = f.size(PLAYER_NAMES[p])
            r = pygame.Rect(0, 0, tw + 8, th)
            r.midbottom = (int(x), int(y))
            while any(r.colliderect(q) for q in placed):
                r.y -= th - 2
            placed.append(r)
            if r.bottom < int(y) - 2:                                 # petit trait vers la tête si le nom est décalé
                D.line(self.screen, (230, 232, 240), (int(x), r.bottom), (int(x), int(y) - 2), 1)
            D.label(self.screen, f, PLAYER_NAMES[p], midbottom=r.midbottom)
            if p == h:
                D.polygon(self.screen, ST.OUTLINE, [(x, r.y - 1), (x - 7, r.y - 10), (x + 7, r.y - 10)])
                D.polygon(self.screen, (255, 255, 255), [(x, r.y - 4), (x - 4, r.y - 9), (x + 4, r.y - 9)])
            mate = self.human_is_imp and p in w.imps and p != h
            if (self.god and p in w.imps) or mate:   # imposteurs (vue debug) ou votre partenaire : pastille rouge
                ST.ocircle(self.screen, ST.RED, (r.right + 8, r.centery), 6, 2)

    # ------------------------------------------------------------------------ HUD
    def panel(self, rect, alpha=None, color=None):
        """Panneau d'interface : aplat clair (ou `color`) cerné de noir."""
        ST.orect(self.screen, color or ST.PANEL, rect, 10, 3)

    def round_button(self, rect, label, sub, active, fill, hover=False, progress=None, big=False):
        """Gros bouton rond cerné de noir. Inactif : gris, texte atténué."""
        cx, cy = rect.center
        r = 50
        col = fill if active else (128, 136, 154)
        if active and hover:
            col = ST.shade(col, 1.08)
        D.circle(self.screen, ST.OUTLINE, (cx, cy + 5), r + 4)                  # ombre portée
        ST.ocircle(self.screen, col, (cx, cy), r, 4)
        if progress:
            D.arc(self.screen, ST.GREEN, (cx - r + 6, cy - r + 6, 2 * r - 12, 2 * r - 12),
                            math.pi / 2 - 2 * math.pi * min(1.0, progress), math.pi / 2, 7)
        ss = 12                                 # le sous-titre doit tenir dans le bouton
        while sub and ss > 9 and self.font(ss, True).size(sub)[0] > 2 * r - 16:
            ss -= 1
        if active and sum(col) > 600:          # bouton clair : texte foncé, sans contour
            self.text(label, (cx, cy - (6 if sub else 0)), 30 if big else 17, ST.INK, True, center=True)
            if sub:
                self.text(sub, (cx, cy + 18), ss, ST.MUTED, True, center=True)
            return
        tc = (255, 255, 255) if active else (205, 210, 222)
        self.otext(label, (cx, cy - (6 if sub else 0)), 30 if big else 17, tc, center=True)
        if sub:
            self.otext(sub, (cx, cy + 18), ss, tc, center=True, width=1)

    def draw_hud(self):
        w, h = self.world, self.human
        # barre de tâches globale
        bar = pygame.Rect(20, 16, 330, 30)
        frac = w.tasks_fraction()
        ST.orect(self.screen, (42, 47, 60), bar, 6, 3)
        if frac > 0:
            D.rect(self.screen, ST.GREEN, (bar.x + 3, bar.y + 3, int((bar.w - 6) * frac), bar.h - 6),
                             border_radius=4)
        self.otext("TÂCHES TERMINÉES", bar.center, 15, center=True)
        y = 66
        if self.human_alive and not self.human_is_imp and w.lights_off:
            self.otext("LUMIÈRES COUPÉES — réparez au panneau d'Électricité (E)", (W // 2, 48), 18, ST.YELLOW,
                       center=True)
        if self.human_alive and not self.human_is_imp:
            rows = sorted(w.tasks[h])
            self.panel(pygame.Rect(20, y - 8, 250, 32 + 22 * max(1, len(rows))))
            self.text("Mes tâches", (34, y - 2), 15, ST.INK, True)
            for r in rows:
                y += 22
                prog = w.work[h].get(r, 0.0) / w.rt.task_time
                self.text(f"• {ROOMS[r]}" + (f"  {int(prog * 100)} %" if prog > 0 else ""), (38, y + 2), 14,
                          (150, 96, 10), True)
            if not rows:
                self.text("Tout est fait !", (38, y + 24), 14, (34, 130, 60), True)
        elif self.human_alive:
            self.panel(pygame.Rect(20, y - 8, 380, 80), color=(250, 222, 222))
            mates = [PLAYER_NAMES[q] + ("" if w.alive[q] else " (éjecté)") for q in w.imps if q != h]
            self.text("IMPOSTEUR" + (f"  —  partenaire : {', '.join(mates)}" if mates else ""), (34, y - 2), 16,
                      (170, 30, 38), True)
            self.text(f"Crewmates en vie : {len(w.crew_alive())}, imposteurs : {len(w.imps_alive())}"
                      f"  (victoire à égalité)", (34, y + 22), 13, ST.INK, True)
            sab = "prêt" if w.can_sabotage() else ("en cours" if w.lights_off else f"dans {int(w.sab_ready_t - w.time) + 1} s")
            self.text(f"L : lumières ({sab})    C : conduit (sur une grille)", (34, y + 44), 12, ST.MUTED, True)
        # chrono + rôle
        remaining = max(0, int(w.rt.max_time - w.time))
        mm = self.mini.get_rect(topright=(W - 20, 58))
        self.otext(f"{remaining // 60}:{remaining % 60:02d}", (W - 22, 12), 26, right=True, width=3)
        if self.human is None and self.free_cam:
            spec = "SPECTATEUR — caméra libre"
        else:
            spec = "SPECTATEUR — suivi : " + PLAYER_NAMES[self.focus_player()]
        label = spec if self.human is None else (
            "FANTÔME — vous voyez tout" if not self.human_alive else "")
        if label:
            self.otext(label, (W // 2, 20), 17, center=True)
        # minicarte
        ST.orect(self.screen, ST.SPACE, mm.inflate(8, 8), 6, 3)
        self.screen.blit(self.mini, mm)
        self._mm_rect = mm
        sc = mm.w / M.WORLD_W
        if self.human is None:                  # cadre de la vue sur la minicarte
            self.screen.set_clip(mm)
            D.rect(self.screen, (255, 255, 255),
                             (mm.x + self.cam[0] * sc, mm.y + self.cam[1] * sc, W * sc, H * sc), 1)
            self.screen.set_clip(None)
        if self.human_alive and not self.human_is_imp:
            for r in w.tasks[h]:
                tx, ty = M.task_terminal(r)
                ST.orect(self.screen, ST.YELLOW, (mm.x + tx * sc - 3, mm.y + ty * sc - 3, 6, 6), 0, 1)
        for p in range(w.n):
            if not w.alive[p] or not (self.full_view() or p == h):
                continue
            c = (int(mm.x + w.x[p] * sc), int(mm.y + w.y[p] * sc))
            ST.ocircle(self.screen, COLORS[p], c, 4, 2)
            if p == h:
                D.circle(self.screen, (255, 255, 255), c, 8, 2)
        self.draw_map_button()
        # journal
        yy = H - 28 - 19 * min(5, len(self.log))
        for msg, col in list(self.log)[-5:]:
            self.otext(msg, (20, yy), 14, col, width=2)
            yy += 19
        # boutons
        if self.human_alive:
            self.draw_buttons()
        # aide
        if self.human is None:
            hint = ("ZQSD / flèches (Maj : vite) ou glisser : caméra   clic minicarte : y aller   Tab : suivre un joueur   "
                    "F : caméra libre   +/- : vitesse   P : pause   M : carte   Échap : menu")
        elif self.human_is_imp:
            hint = "ZQSD : bouger   K : tuer   C : conduit   L : lumières   R : signaler   U : urgence   M : carte   Échap"
        else:
            hint = "ZQSD : bouger   E : tâche / réparer   R : signaler   U : urgence   M : carte   Échap"
        self.otext(hint, (W // 2, H - 10), 11, (205, 212, 228), center=True, width=1)

    def draw_map_button(self):
        r, hover = self.btn_map, self.btn_map.collidepoint(self.mouse_pos())
        ST.orect(self.screen, ST.shade(ST.PANEL, 1.06) if hover or self.show_map else ST.PANEL, r, 10, 3)
        self.text("CARTE  [M]", r.center, 15, ST.INK, True, center=True)

    def draw_report_button(self):
        w = self.world
        mouse = self.mouse_pos()
        can = w.human_can_report()
        self.round_button(self.btn_report, "SIGNALER", "[R]" if can else "aucun corps", can, ST.RED,
                          self.btn_report.collidepoint(mouse))
        # bouton d'urgence (dans la Cafétéria seulement)
        if w.room[self.human] == 0 and w.emergencies_left[self.human] > 0:
            ok = w.human_can_emergency()
            wait = int(math.ceil(w.emergency_wait()))
            self.round_button(self.btn_emergency, "URGENCE", "[U]" if ok else f"dans {wait} s", ok, (232, 90, 60),
                              self.btn_emergency.collidepoint(mouse))

    def draw_buttons(self):
        w, h = self.world, self.human
        mouse = self.mouse_pos()
        self.draw_report_button()
        if self.human_is_imp:
            ready = w.human_can_kill()
            if w.cd_s > 0:
                self.round_button(self.btn_use, str(int(math.ceil(w.cd_s))), "recharge", False, ST.RED, big=True)
            else:
                self.round_button(self.btn_use, "TUER", "[K]" if ready else "trop loin", ready, ST.RED,
                                  self.btn_use.collidepoint(mouse))
            # conduit : prochaine sortie affichée
            can_v = w.human_can_vent()
            if can_v:
                links = M.VENT_LINKS[w.room[h]]
                sub = "vers " + ROOMS[links[w.vent_idx % len(links)]] + " [C]"
            else:
                sub = "sur une grille"
            self.round_button(self.btn_vent, "CONDUIT", sub, can_v, (96, 110, 138), self.btn_vent.collidepoint(mouse))
            # sabotage des lumières
            can_s = w.can_sabotage()
            sub = ("lumières [L]" if can_s else "en cours" if w.lights_off
                   else f"dans {int(w.sab_ready_t - w.time) + 1} s")
            self.round_button(self.btn_sabotage, "SABOTAGE", sub, can_s, (150, 70, 190),
                              self.btn_sabotage.collidepoint(mouse))
            return
        near = w.my_task_nearby(h)
        if near is None and w.near_fix(h):
            near = M.FIX_ROOM
        prog = None
        if near is not None:
            prog = (w.fix_prog / w.rt.fix_time) if w.near_fix(h) else w.work[h].get(near, 0.0) / w.rt.task_time
        self.round_button(self.btn_use, "UTILISER", "[E] maintenir" if near is not None else "approchez",
                          near is not None, (245, 245, 245), self.btn_use.collidepoint(mouse), prog)

    def draw_big_map(self):
        w, h = self.world, self.human
        box = pygame.Rect(40, 40, W - 80, H - 80)
        ST.orect(self.screen, ST.SPACE, box, 12, 4)
        size = (box.w - 30, int((box.w - 30) * M.WORLD_H / M.WORLD_W))
        surf = mini_surface(self.static, size)
        if id(surf) not in self.screen._hires:
            self._register_map(surf)
        pos = surf.get_rect(center=box.center)
        self.screen.blit(surf, pos)
        self._bigmap = (pos, size[0] / M.WORLD_W)
        sc = size[0] / M.WORLD_W
        if self.human_alive and not self.human_is_imp:
            for r in w.tasks[h]:
                tx, ty = M.task_terminal(r)
                ST.orect(self.screen, ST.YELLOW, (pos.x + tx * sc - 6, pos.y + ty * sc - 6, 12, 12), 2, 2)
        for p in range(w.n):
            if not w.alive[p] or not (self.full_view() or p == h):
                continue
            c = (int(pos.x + w.x[p] * sc), int(pos.y + w.y[p] * sc))
            ST.ocircle(self.screen, COLORS[p], c, 8, 3)
            if p == h:
                D.circle(self.screen, (255, 255, 255), c, 13, 2)
        self.otext("CARTE  (M pour fermer" + (" · clic : y aller)" if self.human is None else ")"),
                   (box.centerx, box.y + 16), 18, center=True)

    # ------------------------------------------------------------ boucle principale
    def set_mode(self, mode):
        """mode : 'crew' | 'imposter' | 'spectate'."""
        if mode == "spectate":
            self.human, self.free_cam = None, True
        else:
            self.human, self.human_role = 0, mode
        self.speed = 1.0

    def run(self):
        w = self.world
        acc = 0.0
        if self.menu:
            mode = self.main_menu()
            if mode is None:
                pygame.quit()
                return
            self.set_mode(mode)
            self.new_game()
        while True:
            self.over = False
            while not self.over:
                dt = min(self.clock.tick(60) / 1000.0, 0.1)
                self.anim_t += dt
                for e in self.poll():
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_ESCAPE:
                            if self.menu:                 # retour au menu
                                mode = self.main_menu()
                                if mode is None:
                                    pygame.quit()
                                    return
                                self.set_mode(mode)
                                self.new_game()
                                acc = 0.0
                                break
                            pygame.quit()
                            return
                        if e.key == pygame.K_F11:
                            self.toggle_fullscreen()
                        elif e.key == pygame.K_f and self.human is None:
                            self.free_cam = not self.free_cam
                        if e.key == pygame.K_m:
                            self.show_map = not self.show_map
                        elif e.key == pygame.K_v:
                            self.god = not self.god
                        elif e.key == pygame.K_r and self.human_alive:
                            w.human_report()
                        elif e.key == pygame.K_u and self.human_alive:
                            w.human_emergency()
                        elif e.key == pygame.K_c and self.human_alive:
                            w.human_vent()
                        elif e.key == pygame.K_l and self.human_alive:
                            w.human_sabotage()
                        elif e.key in (pygame.K_k, pygame.K_f) and self.human_alive:
                            w.human_kill()
                        elif e.key == pygame.K_TAB and self.human is None:
                            alive = w.alive_players()
                            if alive:
                                cur = self.follow if self.follow in alive else alive[0]
                                self.follow = alive[(alive.index(cur) + 1) % len(alive)] if not self.free_cam else cur
                                self.free_cam = False
                        elif e.key == pygame.K_p and self.human is None:
                            self.paused = not self.paused
                        elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                            self.speed = min(12.0, self.speed * 1.5)
                        elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.speed = max(0.25, self.speed / 1.5)
                    elif self.human is None and e.type == pygame.MOUSEMOTION and (e.buttons[0] or e.buttons[2]):
                        self.free_cam = True                     # glisser : déplacer la caméra
                        self.cam[0] -= e.rel[0]
                        self.cam[1] -= e.rel[1]
                        self.clamp_camera()
                    elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.btn_map.collidepoint(e.pos):
                        self.show_map = not self.show_map
                    elif self.human is None and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                        if self.show_map and self._bigmap:
                            pos, sc = self._bigmap
                            if pos.collidepoint(e.pos):
                                self.free_cam = True
                                self.center_camera((e.pos[0] - pos.x) / sc, (e.pos[1] - pos.y) / sc)
                                self.show_map = False
                        elif self._mm_rect and self._mm_rect.collidepoint(e.pos):
                            mm = self._mm_rect
                            sc = mm.w / M.WORLD_W
                            self.free_cam = True
                            self.center_camera((e.pos[0] - mm.x) / sc, (e.pos[1] - mm.y) / sc)
                    elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.human_alive:
                        if self.btn_report.collidepoint(e.pos):
                            w.human_report()
                        elif self.btn_emergency.collidepoint(e.pos):
                            w.human_emergency()
                        elif self.human_is_imp and self.btn_use.collidepoint(e.pos):
                            w.human_kill()
                        elif self.human_is_imp and self.btn_vent.collidepoint(e.pos):
                            w.human_vent()
                        elif self.human_is_imp and self.btn_sabotage.collidepoint(e.pos):
                            w.human_sabotage()
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
            res = self.end_screen()
            if res == "menu":
                mode = self.main_menu()
                if mode is None:
                    res = False
                else:
                    self.set_mode(mode)
            if not res:
                pygame.quit()
                return
            self.new_game()
            acc = 0.0

    # ------------------------------------------------------------ menu de démarrage
    def main_menu(self):
        """Choix du mode : 'crew' | 'imposter' | 'spectate', ou None pour quitter."""
        modes = [("crew", "CREWMATE", "Faites vos tâches, trouvez les imposteurs", COLORS[2], "1"),
                 ("imposter", "IMPOSTEUR", "Éliminez les crewmates sans vous faire voir", COLORS[0], "2"),
                 ("spectate", "SPECTATEUR", "Regardez les IA jouer, caméra libre", COLORS[1], "3")]
        rects = [pygame.Rect(W // 2 - 510 + i * 350, 250, 320, 300) for i in range(3)]
        quit_r = pygame.Rect(W // 2 - 110, 600, 220, 52)
        rng = random.Random(1)
        stars = [(rng.randrange(W), rng.randrange(H), rng.choice((1, 1, 2, 3))) for _ in range(160)]
        while True:
            t = pygame.time.get_ticks() / 1000.0
            mouse = self.mouse_pos()
            self.screen.fill(ST.BLUE_BG)
            self.screen.blit(meeting_background(self.screen), (0, 0))
            for sx, sy, sz in stars:
                D.circle(self.screen, (200, 205, 230), (int((sx - t * 12 * sz) % W), sy), 1 if sz < 3 else 2)
            self.otext("AMONG US IA", (W // 2, 110), 64, center=True, width=4)
            self.text("projet de fan · 5 crewmates, 2 imposteurs, IA entraînées par MAPPO", (W // 2, 172), 18,
                      (205, 212, 228), True, center=True)
            for (mode, title, sub, col, key), r in zip(modes, rects):
                hover = r.collidepoint(mouse)
                ST.orect(self.screen, (255, 252, 236) if hover else ST.CARD, r.inflate(10, 10) if hover else r, 16,
                         5 if hover else 4)
                bob = math.sin(t * 3) * 5 if hover else 0
                spr = D.sprite(140, 160, 1.7, lambda c: scene.draw_bot(c, 70, 130, col, 1, moving=hover, t=t),
                               self.screen.k)
                self.screen.blit_native(spr, midbottom=(r.centerx, r.y + 172 + bob))
                self.otext(title, (r.centerx, r.y + 192), 28, center=True, width=3)
                for i, ln in enumerate(self.wrap(sub, 15, r.w - 40, True)):
                    self.text(ln, (r.centerx, r.y + 232 + 19 * i), 15, ST.MUTED, True, center=True)
                self.text(f"[{key}]", (r.centerx, r.bottom - 22), 14, ST.MUTED, True, center=True)
            self.big_button(quit_r, "QUITTER  [Échap]", (135, 147, 166), True, quit_r.collidepoint(mouse))
            self.text("F11 : plein écran / fenêtre", (W // 2, H - 30), 14, (170, 178, 198), True, center=True)
            pygame.display.flip()
            self.clock.tick(60)
            for e in self.poll():
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for (mode, *_), r in zip(modes, rects):
                        if r.collidepoint(e.pos):
                            return mode
                    if quit_r.collidepoint(e.pos):
                        return None
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return None
                    if e.key == pygame.K_F11:
                        self.toggle_fullscreen()
                    for i, (mode, *_) in enumerate(modes):
                        if e.key in (pygame.K_1 + i, pygame.K_KP1 + i):
                            return mode
