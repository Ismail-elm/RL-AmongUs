"""Rendu « vue de dessus » : décor des salles, portes, personnages (petits robots originaux)."""
from __future__ import annotations

import math

import pygame

from . import canvas as D

from .mapdata import ADJ, LAYOUT, ROOMS

WALL = 34
DOOR = 92

# (couleur 1, couleur 2, motif, accessoires, position de la console de tâche)
# accessoires : (type, fx, fy[, option]) en fraction de la surface de sol
SPEC = {
    0: dict(floor=((170, 172, 160), (156, 158, 148)), pat="diamond",
            props=[("table", .22, .27), ("table", .78, .27, "button")], spot=None),
    1: dict(floor=((72, 80, 96), (64, 72, 88)), pat="grid",
            props=[("turret", .18, .42), ("console", .50, .06), ("console", .80, .06)], spot=(.80, .46)),
    2: dict(floor=((58, 74, 100), (52, 66, 92)), pat="grid",
            props=[("monitors", .50, .06), ("console", .25, .30), ("console", .75, .30)], spot=(.50, .34)),
    3: dict(floor=((70, 104, 92), (62, 94, 84)), pat="plate",
            props=[("plant", .15, .30), ("plant", .85, .32), ("tank", .50, .70)], spot=(.28, .62)),
    4: dict(floor=((82, 70, 104), (74, 62, 96)), pat="grid",
            props=[("panel", .20, .06), ("panel", .50, .06), ("panel", .80, .06), ("holo", .50, .66)],
            spot=(.50, .34)),
    5: dict(floor=((112, 98, 76), (104, 90, 70)), pat="plate",
            props=[("crate", .12, .20), ("crate", .22, .20), ("crate", .12, .34), ("crate", .84, .70),
                   ("crate", .72, .70), ("crate", .84, .84), ("crate", .50, .16)], spot=(.60, .46)),
    6: dict(floor=((92, 104, 112), (84, 96, 104)), pat="diamond",
            props=[("holo", .50, .50), ("console", .22, .08), ("console", .78, .08)], spot=(.24, .60)),
    7: dict(floor=((96, 92, 70), (88, 84, 64)), pat="plate",
            props=[("panel", .16, .06), ("panel", .38, .06), ("panel", .62, .06), ("panel", .84, .06)],
            spot=(.50, .30)),
    8: dict(floor=((74, 78, 82), (66, 70, 74)), pat="plate",
            props=[("cyl", .20, .70), ("cyl", .50, .70), ("cyl", .80, .70), ("console", .50, .06)],
            spot=(.50, .34)),
    9: dict(floor=((96, 70, 66), (88, 62, 58)), pat="grid",
            props=[("reactor", .70, .50), ("panel", .18, .06)], spot=(.28, .46)),
    10: dict(floor=((60, 68, 78), (54, 62, 72)), pat="grid",
             props=[("monitors", .30, .06), ("monitors", .72, .06), ("console", .50, .50)], spot=(.50, .30)),
    11: dict(floor=((150, 170, 176), (138, 158, 166)), pat="diamond",
             props=[("bed", .18, .30), ("bed", .18, .62), ("scanner", .72, .46)], spot=(.74, .72)),
}

_bg_cache: dict = {}


def _pt(fr: pygame.Rect, fx: float, fy: float):
    return int(fr.x + fx * fr.w), int(fr.y + fy * fr.h)


def floor_rect(view: pygame.Rect) -> pygame.Rect:
    return view.inflate(-2 * WALL, -2 * WALL)


def task_spot(r: int, view: pygame.Rect):
    sp = SPEC[r]["spot"]
    return None if sp is None else _pt(floor_rect(view), *sp)


def idle_slot(view: pygame.Rect, i: int):
    fr = floor_rect(view)
    return _pt(fr, 0.22 + 0.19 * (i % 4), 0.56 + 0.17 * (i // 4))


# ---------------------------------------------------------------------- portes
def door_layout(r: int, view: pygame.Rect):
    """Une porte par salle voisine, placée du côté où se trouve cette salle sur la carte."""
    fr = floor_rect(view)
    cx0, cy0 = LAYOUT[r]
    by_side = {"N": [], "S": [], "E": [], "W": []}
    for idx, nb in enumerate(sorted(ADJ[r])):
        dx, dy = (LAYOUT[nb][0] - cx0) * 642, (LAYOUT[nb][1] - cy0) * 508
        if abs(dx) > abs(dy):
            side = "E" if dx > 0 else "W"
            key = dy
        else:
            side = "S" if dy > 0 else "N"
            key = dx
        by_side[side].append((key, nb, idx + 1))
    doors = []
    for side, items in by_side.items():
        items.sort()
        k = len(items)
        for i, (_, nb, num) in enumerate(items):
            f = (i + 1) / (k + 1)
            if side in "NS":
                cx = int(fr.x + f * fr.w)
                if side == "N":
                    gap = pygame.Rect(cx - DOOR // 2, view.y, DOOR, WALL + 8)
                    inp, outp = (cx, fr.y + 26), (cx, view.y + 8)
                else:
                    gap = pygame.Rect(cx - DOOR // 2, fr.bottom - 8, DOOR, WALL + 8)
                    inp, outp = (cx, fr.bottom - 26), (cx, view.bottom - 8)
            else:
                cy = int(fr.y + f * fr.h)
                if side == "W":
                    gap = pygame.Rect(view.x, cy - DOOR // 2, WALL + 8, DOOR)
                    inp, outp = (fr.x + 26, cy), (view.x + 8, cy)
                else:
                    gap = pygame.Rect(fr.right - 8, cy - DOOR // 2, WALL + 8, DOOR)
                    inp, outp = (fr.right - 26, cy), (view.right - 8, cy)
            doors.append(dict(room=nb, side=side, rect=gap, in_pos=inp, out_pos=outp, num=num))
    return doors


# ----------------------------------------------------------------------- décor (2.5D)
def _shadow(surf, x, y, rx, ry, a=80):
    tmp = pygame.Surface((int(2 * rx), int(2 * ry)), pygame.SRCALPHA)
    D.ellipse(tmp, (0, 0, 0, a), tmp.get_rect())
    surf.blit(tmp, (x - rx, y - ry))


def _cyl(surf, x, y, rx, ry, h, top, side, rim=None):
    """Cylindre vu de 3/4 : (x, y) = centre de la face supérieure."""
    D.ellipse(surf, _shade(side, 0.7), (x - rx, y + h - ry, 2 * rx, 2 * ry))
    D.rect(surf, side, (x - rx, y, 2 * rx, h))
    D.rect(surf, _shade(side, 1.25), (x - rx, y, max(3, rx // 3), h))          # reflet latéral
    D.rect(surf, _shade(side, 0.78), (x + rx - max(3, rx // 3), y, max(3, rx // 3), h))
    D.ellipse(surf, _shade(side, 0.9), (x - rx, y + h - ry, 2 * rx, 2 * ry), 0) if False else None
    D.ellipse(surf, top, (x - rx, y - ry, 2 * rx, 2 * ry))
    D.ellipse(surf, rim or _shade(top, 1.25), (x - rx, y - ry, 2 * rx, 2 * ry), 2)


def _box(surf, x0, y0, w, h, depth, top, front, edge=None):
    """Pavé vu de 3/4 : (x0, y0) = coin supérieur gauche de la face avant ; la face dessus est au-dessus."""
    D.rect(surf, _shade(front, 0.7), (x0 + 2, y0 + 3, w, h), border_radius=4)
    D.rect(surf, front, (x0, y0, w, h), border_radius=4)
    D.rect(surf, _shade(front, 0.78), (x0, y0 + h - 8, w, 8), border_bottom_left_radius=4,
                     border_bottom_right_radius=4)
    D.polygon(surf, top, [(x0 + 6, y0 - depth), (x0 + w - 6, y0 - depth), (x0 + w, y0), (x0, y0)])
    D.line(surf, edge or _shade(top, 1.3), (x0 + 6, y0 - depth), (x0 + w - 6, y0 - depth), 2)
    D.line(surf, _shade(front, 1.3), (x0, y0 + 1), (x0 + w, y0 + 1), 1)


FLAT = True          # style cartoon : pas de halos lumineux sur les accessoires


def _glow(surf, x, y, r, color, a=70):
    if FLAT:
        return
    tmp = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    for i in range(6):
        rr = int(r * (1 - i / 6))
        D.circle(tmp, (*color, int(a * (i + 1) / 6 * 0.5)), (r, r), rr)
    surf.blit(tmp, (x - r, y - r))


def _prop(surf, kind, x, y, opt=None):
    if kind == "table":
        _shadow(surf, x, y + 52, 96, 30, 90)
        for dx, dy in ((-70, 8), (70, 8), (0, -44), (0, 62)):
            _shadow(surf, x + dx, y + dy + 14, 20, 7, 80)
            _cyl(surf, x + dx, y + dy, 15, 6, 10, (60, 120, 180), (34, 78, 126))
        _cyl(surf, x, y, 58, 36, 16, (88, 160, 214), (38, 88, 140), rim=(160, 214, 250))
        D.ellipse(surf, (110, 178, 226), (x - 36, y - 24, 40, 20))
        if opt == "button":
            _cyl(surf, x, y - 2, 20, 10, 8, (240, 200, 60), (170, 130, 30))
            _cyl(surf, x, y - 8, 8, 4, 6, (230, 70, 60), (160, 40, 40))
    elif kind == "console":
        _shadow(surf, x, y + 44, 64, 12, 80)
        _box(surf, x - 52, y + 10, 104, 34, 20, (82, 92, 116), (36, 42, 58))
        for i in range(3):
            D.rect(surf, (60, 210, 140), (x - 44 + i * 31, y + 17, 26, 14), border_radius=3)
            D.line(surf, (190, 255, 220), (x - 42 + i * 31, y + 19), (x - 24 + i * 31, y + 19), 1)
            _glow(surf, x - 31 + i * 31, y + 24, 26, (80, 230, 160), 50)
    elif kind == "monitors":
        _shadow(surf, x, y + 42, 100, 12, 80)
        _box(surf, x - 92, y + 6, 184, 34, 12, (64, 72, 94), (28, 32, 44))
        for i in range(4):
            sx = x - 84 + i * 43
            D.rect(surf, (16, 20, 28), (sx - 2, y - 24, 42, 38), border_radius=4)
            D.rect(surf, (60, 140, 216), (sx, y - 22, 38, 30), border_radius=3)
            D.rect(surf, (120, 200, 255), (sx, y - 22, 38, 12), border_radius=3)
            D.line(surf, (230, 250, 255), (sx + 4, y - 4), (sx + 24, y - 14), 2)
            _glow(surf, sx + 19, y - 8, 30, (90, 170, 255), 46)
    elif kind == "panel":
        _shadow(surf, x, y + 52, 48, 8, 70)
        _box(surf, x - 38, y + 4, 76, 46, 12, (70, 78, 100), (34, 38, 52))
        for i, c in enumerate(((230, 70, 70), (240, 210, 70), (70, 190, 240), (90, 220, 120))):
            D.line(surf, c, (x - 28 + i * 18, y + 12), (x - 28 + i * 18, y + 38), 3)
            D.circle(surf, _shade(c, 1.3), (x - 28 + i * 18, y + 42), 3)
    elif kind == "crate":
        _shadow(surf, x + 4, y + 30, 36, 10, 90)
        _box(surf, x - 26, y - 14, 52, 40, 16, (176, 132, 84), (110, 78, 48))
        D.line(surf, (70, 50, 30), (x - 26, y + 6), (x + 26, y + 6), 3)
        D.line(surf, (70, 50, 30), (x, y - 14), (x, y + 26), 3)
        D.polygon(surf, (150, 112, 70), [(x - 6, y - 30), (x + 6, y - 30), (x + 6, y - 14), (x - 6, y - 14)])
    elif kind == "cyl":
        _shadow(surf, x + 6, y + 40, 56, 16, 90)
        _cyl(surf, x, y - 6, 42, 17, 40, (168, 176, 188), (110, 118, 130))
        D.ellipse(surf, (200, 206, 216), (x - 28, y - 16, 56, 20), 3)
        D.ellipse(surf, (96, 102, 112), (x - 12, y - 11, 24, 10))
    elif kind == "tank":
        _shadow(surf, x + 8, y + 52, 70, 20, 90)
        _cyl(surf, x, y - 20, 52, 20, 70, (130, 214, 228), (56, 140, 158), rim=(210, 246, 252))
        D.rect(surf, (190, 240, 250), (x - 34, y - 14, 8, 62), border_radius=4)
        D.ellipse(surf, (230, 250, 255), (x - 22, y - 30, 24, 10))
    elif kind == "plant":
        _shadow(surf, x + 4, y + 30, 40, 12, 90)
        _cyl(surf, x, y + 8, 22, 9, 22, (120, 82, 56), (86, 56, 38))
        for dx, dy, rr in ((0, -10, 24), (-18, -2, 17), (18, -2, 17), (0, -30, 15)):
            D.circle(surf, (50, 140, 76), (x + dx, y + dy + 3), rr)
            D.circle(surf, (84, 184, 100), (x + dx, y + dy), rr)
        D.circle(surf, (140, 226, 140), (x - 8, y - 18), 8)
    elif kind == "bed":
        _shadow(surf, x + 4, y + 36, 76, 14, 90)
        _box(surf, x - 66, y - 4, 132, 36, 22, (232, 240, 246), (150, 180, 204))
        D.rect(surf, (200, 222, 240), (x - 62, y - 22, 38, 16), border_radius=7)
        D.rect(surf, (110, 166, 200), (x - 20, y - 22, 80, 20), border_radius=5)
    elif kind == "scanner":
        _shadow(surf, x + 6, y + 56, 82, 22, 90)
        _cyl(surf, x, y - 4, 62, 26, 24, (206, 224, 232), (120, 146, 160))
        D.ellipse(surf, (80, 200, 230), (x - 44, y - 18, 88, 28), 5)
        _glow(surf, x, y - 4, 60, (120, 230, 250), 60)
        D.ellipse(surf, (170, 240, 255), (x - 22, y - 11, 44, 14), 3)
    elif kind == "reactor":
        _shadow(surf, x + 8, y + 84, 120, 34, 100)
        _cyl(surf, x, y, 96, 36, 38, (150, 96, 80), (96, 56, 48))
        _cyl(surf, x, y - 8, 66, 25, 12, (206, 112, 66), (140, 70, 44))
        _glow(surf, x, y - 10, 90, (255, 150, 60), 90)
        D.ellipse(surf, (255, 190, 90), (x - 40, y - 24, 80, 30))
        D.ellipse(surf, (255, 244, 180), (x - 18, y - 17, 36, 14))
    elif kind == "holo":
        _shadow(surf, x + 4, y + 40, 68, 18, 90)
        _cyl(surf, x, y + 4, 58, 22, 20, (92, 124, 154), (44, 62, 84))
        D.ellipse(surf, (110, 220, 250), (x - 44, y - 12, 88, 32), 3)
        _glow(surf, x, y - 20, 56, (110, 220, 250), 60)
        D.polygon(surf, (140, 236, 255), [(x, y - 52), (x + 24, y - 8), (x - 24, y - 8)], 3)
    elif kind == "turret":
        _shadow(surf, x + 6, y + 46, 64, 18, 90)
        _cyl(surf, x, y, 48, 19, 26, (118, 128, 142), (66, 74, 88))
        D.rect(surf, (46, 52, 64), (x - 10, y - 74, 20, 70), border_radius=6)
        D.rect(surf, (96, 104, 120), (x - 10, y - 74, 7, 70), border_radius=4)
        D.circle(surf, (226, 90, 80), (x, y - 4), 13)
        D.circle(surf, (255, 170, 160), (x - 4, y - 8), 5)


def _floor_pattern(surf, fr, c1, c2, pat):
    D.rect(surf, c1, fr, border_radius=10)
    clip = surf.get_clip()
    surf.set_clip(fr)
    if pat == "diamond":
        s = 44
        for gy in range(-1, fr.h // s + 2):
            for gx in range(-1, fr.w // (s * 2) + 2):
                cx = fr.x + gx * s * 2 + (s if gy % 2 else 0)
                cy = fr.y + gy * s
                D.polygon(surf, c2, [(cx, cy - s // 2), (cx + s, cy), (cx, cy + s // 2), (cx - s, cy)])
    elif pat == "grid":
        for x in range(fr.x, fr.right, 48):
            D.line(surf, c2, (x, fr.y), (x, fr.bottom), 2)
        for y in range(fr.y, fr.bottom, 48):
            D.line(surf, c2, (fr.x, y), (fr.right, y), 2)
    elif pat == "plate":
        for y in range(fr.y, fr.bottom, 56):
            D.line(surf, c2, (fr.x, y), (fr.right, y), 3)
            for x in range(fr.x + 28 + (28 if (y // 56) % 2 else 0), fr.right, 56):
                D.circle(surf, c2, (x, y + 28), 3)
    surf.set_clip(clip)


def room_background(r: int, view: pygame.Rect, font) -> pygame.Surface:
    """Décor statique d'une salle (mis en cache) : murs, sol, accessoires, portes."""
    key = (r, view.size)
    if key in _bg_cache:
        return _bg_cache[key]
    surf = pygame.Surface(view.size)
    local = pygame.Rect(0, 0, *view.size)
    surf.fill((17, 20, 33))
    spec = SPEC[r]
    D.rect(surf, (12, 14, 20), local, border_radius=28)
    D.rect(surf, (52, 58, 74), local.inflate(-6, -6), border_radius=26)
    D.rect(surf, (36, 41, 54), local.inflate(-18, -18), border_radius=22)
    fr = floor_rect(local)
    _floor_pattern(surf, fr, spec["floor"][0], spec["floor"][1], spec["pat"])
    D.rect(surf, (20, 24, 32), fr, 4, border_radius=10)
    for p in spec["props"]:
        kind, fx, fy = p[0], p[1], p[2]
        _prop(surf, kind, *_pt(fr, fx, fy), p[3] if len(p) > 3 else None)
    if spec["spot"] is not None:
        sx, sy = _pt(fr, *spec["spot"])
        D.rect(surf, (24, 28, 38), (sx - 26, sy - 20, 52, 40), border_radius=7)
        D.rect(surf, (70, 200, 220), (sx - 20, sy - 15, 40, 24), border_radius=4)
        D.rect(surf, (90, 96, 112), (sx - 26, sy - 20, 52, 40), 2, border_radius=7)
    # portes
    for d in door_layout(r, local):
        g = d["rect"]
        D.rect(surf, (28, 32, 42), g)
        inner = g.inflate(-8, -8)
        D.rect(surf, tuple(c - 14 for c in spec["floor"][0]), inner)
        D.rect(surf, (200, 206, 220), g, 3)
        # flèche vers l'extérieur
        cx, cy = g.center
        ax, ay = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}[d["side"]]
        px, py = -ay, ax
        tip = (cx + ax * 14, cy + ay * 14)
        D.polygon(surf, (240, 200, 80), [tip, (cx - ax * 8 + px * 12, cy - ay * 8 + py * 12),
                                                    (cx - ax * 8 - px * 12, cy - ay * 8 - py * 12)])
        # étiquette
        label = f"{d['num']}  {ROOMS[d['room']]}"
        ts = font.render(label, True, (240, 242, 250))
        chip = ts.get_rect().inflate(14, 8)
        if d["side"] == "N":
            chip.midtop = (cx, fr.y + 6)
        elif d["side"] == "S":
            chip.midbottom = (cx, fr.bottom - 6)
        elif d["side"] == "W":
            chip.midleft = (fr.x + 6, cy)
        else:
            chip.midright = (fr.right - 6, cy)
        D.rect(surf, (20, 24, 36), chip, border_radius=8)
        D.rect(surf, (90, 100, 130), chip, 1, border_radius=8)
        surf.blit(ts, ts.get_rect(center=chip.center))
    _bg_cache[key] = surf
    return surf


# ------------------------------------------------------------------ personnages
def _shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def draw_bot(surf, x, y, color, face=1, moving=False, t=0.0, dead=False, working=False,
             label=None, font=None, you=False, selected=False):
    """Petit robot original, style cartoon : aplats, contour noir épais, visière, antenne (x, y = pieds)."""
    from . import style as S
    x, y = int(x), int(y)
    dark = _shade(color, 0.68)
    sh = pygame.Surface((54, 16), pygame.SRCALPHA)
    D.ellipse(sh, (0, 0, 0, 90), sh.get_rect())
    surf.blit(sh, (x - 27, y - 9))
    if dead:
        # couché sur le côté, yeux en croix
        S.orect(surf, dark, (x + 12, y - 15, 11, 7), 3, 2)
        S.orect(surf, dark, (x + 12, y - 6, 11, 7), 3, 2)
        S.orect(surf, color, (x - 24, y - 22, 38, 22), 10, 3)
        D.rect(surf, dark, (x - 24, y - 9, 38, 9), border_bottom_left_radius=10, border_bottom_right_radius=10)
        S.orect(surf, S.CYAN, (x - 21, y - 19, 13, 14), 5, 2)
        for dx in (-17, -12):
            D.line(surf, S.OUTLINE, (dx + x - 2, y - 15), (dx + x + 2, y - 10), 2)
            D.line(surf, S.OUTLINE, (dx + x - 2, y - 10), (dx + x + 2, y - 15), 2)
        D.line(surf, S.OUTLINE, (x - 26, y - 12), (x - 35, y - 17), 3)
        S.ocircle(surf, S.YELLOW, (x - 36, y - 18), 3, 2)
        if label and font:
            D.label(surf, font, label, (255, 170, 170), midbottom=(x, y - 26))
        return
    bob = int(math.sin(t * 14) * 2.5) if moving else 0
    # sac à dos (derrière, du côté opposé au regard)
    S.orect(surf, dark, (x - face * 17 - 5, y - 31, 10, 20), 4, 3)
    # jambes
    for dx, o in ((-11, bob), (3, -bob)):
        S.orect(surf, dark, (x + dx, y - 10 + o, 9, 10), 3, 2)
    # corps : aplat + bande d'ombre en bas, contour épais
    S.orect(surf, color, (x - 15, y - 38, 30, 32), 12, 3)
    D.rect(surf, dark, (x - 15, y - 15, 30, 9), border_bottom_left_radius=12, border_bottom_right_radius=12)
    # bras
    arm_y = y - 30 if working else y - 20
    for side in (-1, 1):
        S.ocircle(surf, dark, (x + side * 17, arm_y + bob), 4, 2)
    # visière (regarde vers `face`)
    vx = x - 11 + 3 * face
    S.orect(surf, S.CYAN, (vx, y - 33, 22, 12), 6, 2)
    D.rect(surf, (255, 255, 255), (vx + 4, y - 31, 7, 4), border_radius=2)
    # antenne
    D.line(surf, S.OUTLINE, (x, y - 40), (x, y - 49), 3)
    S.ocircle(surf, S.YELLOW, (x, y - 51), 4, 2)
    if working:
        D.circle(surf, S.YELLOW, (x, y - 51), 9, 2)
    if selected:
        D.circle(surf, S.RED, (x, y - 22), 30, 3)
    if label and font:
        r = D.label(surf, font, label, midbottom=(x, y - 58))
        if you:
            D.polygon(surf, S.OUTLINE, [(x, r.y - 1), (x - 7, r.y - 10), (x + 7, r.y - 10)])
            D.polygon(surf, (255, 255, 255), [(x, r.y - 4), (x - 4, r.y - 9), (x + 4, r.y - 9)])
