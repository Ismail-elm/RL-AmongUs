"""Carte CONTINUE du vaisseau (pixels) : salles, couloirs, collisions, vision, navigation.

Le graphe de 12 salles (mapdata.EDGES) est conservé : les mêmes salles sont reliées par de vrais couloirs.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .mapdata import EDGES, N_ROOMS
from . import scene

WORLD_W, WORLD_H = 4000, 2300
CELL = 10          # résolution des masques de collision / vision
NAV = 20           # résolution des champs de navigation
PLAYER_R = 14
GW, GH = WORLD_W // CELL, WORLD_H // CELL
NW, NH = WORLD_W // NAV, WORLD_H // NAV

# (x, y, largeur, hauteur)
ROOM_RECTS = {
    0: (1300, 100, 640, 420),    # Cafétéria
    1: (2400, 100, 520, 340),    # Armes
    2: (3300, 420, 520, 340),    # Navigation
    3: (2400, 700, 520, 340),    # O2
    4: (3000, 1350, 520, 340),   # Boucliers
    5: (1500, 1650, 620, 360),   # Stockage
    6: (1800, 1000, 480, 320),   # Admin
    7: (900, 1750, 520, 340),    # Électricité
    8: (150, 1450, 520, 340),    # Moteurs
    9: (150, 700, 520, 340),     # Réacteur
    10: (800, 1000, 480, 320),   # Sécurité
    11: (520, 150, 640, 340),    # Infirmerie
}

CORRIDORS = {
    (0, 1): [(1930, 210, 480, 120)],
    (0, 11): [(1150, 210, 160, 120)],
    (0, 6): [(1810, 510, 120, 500)],
    (0, 5): [(1520, 510, 120, 1150)],
    (1, 3): [(2560, 430, 120, 280)],
    (1, 2): [(2910, 190, 650, 120), (3500, 190, 120, 240)],
    (3, 2): [(2800, 500, 510, 120), (2800, 500, 120, 210)],
    (3, 4): [(2910, 810, 410, 120), (3200, 810, 120, 550)],
    (2, 4): [(3400, 750, 120, 610)],
    (4, 5): [(2110, 1770, 1210, 120), (3200, 1680, 120, 210)],
    (5, 6): [(1840, 1310, 120, 350)],
    (5, 7): [(1410, 1830, 100, 120)],
    (7, 8): [(320, 1870, 590, 120), (300, 1780, 120, 210)],
    (8, 9): [(250, 1030, 120, 430)],
    (8, 10): [(660, 1490, 440, 120), (980, 1310, 120, 300)],
    (9, 10): [(660, 790, 270, 120), (800, 840, 120, 170)],
    (10, 11): [(1020, 480, 120, 530)],
}


def room_center(r: int):
    x, y, w, h = ROOM_RECTS[r]
    return x + w / 2, y + h / 2


def task_terminal(r: int):
    """Position de la borne de tâche (None si la salle n'a pas de tâche)."""
    sp = scene.SPEC[r]["spot"]
    if sp is None:
        return None
    x, y, w, h = ROOM_RECTS[r]
    return x + sp[0] * w, y + sp[1] * h


def task_stand(r: int):
    t = task_terminal(r)
    return None if t is None else (t[0], t[1] + 56)


SPAWN_CENTER = room_center(0)


def spawn_points(n: int):
    cx, cy = SPAWN_CENTER
    return [(cx + 150 * math.cos(2 * math.pi * i / n), cy + 105 * math.sin(2 * math.pi * i / n)) for i in range(n)]


# ----------------------------------------------------------------- masques
def _rasterize():
    walk = np.zeros((GH, GW), dtype=bool)
    room = np.full((GH, GW), -1, dtype=np.int8)

    def fill(arr, rect, val):
        x, y, w, h = rect
        arr[y // CELL:(y + h) // CELL, x // CELL:(x + w) // CELL] = val

    for rect in CORRIDORS.values():
        for rc in rect:
            fill(walk, rc, True)
    for r, rc in ROOM_RECTS.items():
        fill(walk, rc, True)
        fill(room, rc, r)
    return walk, room


def _erode(mask, iters):
    m = mask.copy()
    for _ in range(iters):
        p = np.pad(m, 1, constant_values=False)
        out = np.ones_like(m)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                out &= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
        m = out
    return m


WALK10, ROOM10 = _rasterize()
WALL_H, CORR_H = 64, 44        # hauteur apparente des murs du fond (rendu 2.5D)


def _vision_mask():
    """Zone « éclairable » : sol + faces avant des murs du fond (même géométrie que le rendu)."""
    vis = WALK10.copy()
    for rc in ROOM_RECTS.values():
        x, y, w, h = rc
        vis[(y - WALL_H) // CELL:y // CELL, (x - 20) // CELL:(x + w + 20) // CELL] = True
    for rects in CORRIDORS.values():
        for x, y, w, h in rects:
            if w >= h:
                vis[(y - CORR_H) // CELL:y // CELL, x // CELL:(x + w) // CELL] = True
    return vis


VIS10 = _vision_mask()
OK10 = _erode(WALK10, 2)                       # centre du joueur autorisé
_WALK_B = bytearray(WALK10.ravel().astype(np.uint8))
_OK_B = bytearray(OK10.ravel().astype(np.uint8))
_ROOM_L = ROOM10.ravel().tolist()
OK20 = _erode(WALK10, 3).reshape(NH, 2, NW, 2).all(axis=(1, 3))   # cellules de navigation bien à l'intérieur
_OK20 = OK20.ravel().tolist()


def can_stand(x: float, y: float) -> bool:
    if x < 0 or y < 0 or x >= WORLD_W or y >= WORLD_H:
        return False
    return _OK_B[int(y) // CELL * GW + int(x) // CELL] == 1


def room_at(x: float, y: float) -> int:
    """Indice de la salle contenant le point, ou -2 (couloir / hors carte)."""
    if x < 0 or y < 0 or x >= WORLD_W or y >= WORLD_H:
        return -2
    v = _ROOM_L[int(y) // CELL * GW + int(x) // CELL]
    return -2 if v < 0 else v


def nearest_room(x: float, y: float) -> int:
    best, bd = 0, 1e18
    for r in range(N_ROOMS):
        cx, cy = room_center(r)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < bd:
            best, bd = r, d
    return best


def inside_room(x: float, y: float, r: int, margin: float = 0.0) -> bool:
    rx, ry, rw, rh = ROOM_RECTS[r]
    return rx + margin <= x <= rx + rw - margin and ry + margin <= y <= ry + rh - margin


def line_of_sight(x1, y1, x2, y2) -> bool:
    d = math.hypot(x2 - x1, y2 - y1)
    n = max(1, int(d // 8))
    for i in range(1, n + 1):
        t = i / n
        x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        if x < 0 or y < 0 or x >= WORLD_W or y >= WORLD_H or _WALK_B[int(y) // CELL * GW + int(x) // CELL] == 0:
            return False
    return True


# ------------------------------------------------------------ vision (raycast)
def visibility_polygon(x: float, y: float, radius: float, n_rays: int = 360):
    """Polygone (monde) visible depuis (x, y) : rayons + raffinement dichotomique (bords nets)."""
    step = 8.0
    ang = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
    cs, sn = np.cos(ang), np.sin(ang)
    ks = np.arange(1, int(radius // step) + 1) * step

    def blocked_at(d):
        px, py = x + cs * d, y + sn * d
        ix = np.clip((px // CELL).astype(int), 0, GW - 1)
        iy = np.clip((py // CELL).astype(int), 0, GH - 1)
        ok = (px >= 0) & (py >= 0) & (px < WORLD_W) & (py < WORLD_H)
        return ~(VIS10[iy, ix] & ok)

    px = x + cs[:, None] * ks[None, :]
    py = y + sn[:, None] * ks[None, :]
    ix = np.clip((px // CELL).astype(int), 0, GW - 1)
    iy = np.clip((py // CELL).astype(int), 0, GH - 1)
    inside = (px >= 0) & (py >= 0) & (px < WORLD_W) & (py < WORLD_H)
    blocked = ~(VIS10[iy, ix] & inside)
    hit = blocked.any(axis=1)
    first = np.where(hit, blocked.argmax(axis=1), len(ks))
    hi = np.minimum(first * step + step, radius)
    lo = np.maximum(first * step, 0.0)
    for _ in range(5):
        mid = (lo + hi) / 2
        b = blocked_at(mid)
        hi = np.where(b, mid, hi)
        lo = np.where(b, lo, mid)
    dist = np.where(hit, hi, radius)
    return np.stack([x + cs * dist, y + sn * dist], axis=1)


# ------------------------------------------------------------------ navigation
_fields: dict = {}
_DIRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


def _snap(x, y):
    cx, cy = int(x) // NAV, int(y) // NAV
    if _OK20[cy * NW + cx]:
        return cx, cy
    for rad in range(1, 6):
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < NW and 0 <= ny < NH and _OK20[ny * NW + nx]:
                    return nx, ny
    return cx, cy


def goal_point(goal):
    kind, r = goal
    return task_stand(r) if kind == "spot" else room_center(r)


def field(goal):
    """Champ de distance (Dijkstra) vers un objectif ('room', r) ou ('spot', r)."""
    if goal in _fields:
        return _fields[goal]
    gx, gy = goal_point(goal)
    sx, sy = _snap(gx, gy)
    inf = float("inf")
    dist = [inf] * (NW * NH)
    dist[sy * NW + sx] = 0.0
    heap = [(0.0, sx, sy)]
    while heap:
        d, cx, cy = heapq.heappop(heap)
        if d > dist[cy * NW + cx]:
            continue
        for dx, dy, c in _DIRS:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= NW or ny >= NH:
                continue
            i = ny * NW + nx
            if not _OK20[i]:
                continue
            if dx and dy and not (_OK20[cy * NW + nx] and _OK20[ny * NW + cx]):
                continue   # pas de coupe d'angle
            nd = d + c
            if nd < dist[i]:
                dist[i] = nd
                heapq.heappush(heap, (nd, nx, ny))
    _fields[goal] = dist
    return dist


def steer(dist, x, y):
    """Vecteur (dx, dy) vers la cellule voisine la plus proche de l'objectif (2 pas d'avance)."""
    cx, cy = int(x) // NAV, int(y) // NAV
    cx = min(max(cx, 0), NW - 1)
    cy = min(max(cy, 0), NH - 1)
    tx, ty = cx, cy
    for _ in range(2):
        best = dist[ty * NW + tx]
        bx, by = tx, ty
        for dx, dy, _c in _DIRS:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < NW and 0 <= ny < NH:
                d = dist[ny * NW + nx]
                if d < best:
                    best, bx, by = d, nx, ny
        tx, ty = bx, by
    return (tx + 0.5) * NAV - x, (ty + 0.5) * NAV - y


def components_topology():
    """Vérifie la topologie : quelles paires de salles sont reliées par un couloir sans passer par une 3e salle."""
    from collections import deque
    seen = np.zeros((GH, GW), dtype=bool)
    adj = set()
    for r in range(N_ROOMS):
        x, y, w, h = ROOM_RECTS[r]
        sx, sy = (x + w // 2) // CELL, (y + h // 2) // CELL
        if seen[sy, sx]:
            continue
    # BFS depuis chaque salle à travers les cases de couloir uniquement
    for r in range(N_ROOMS):
        x, y, w, h = ROOM_RECTS[r]
        q = deque()
        vis = np.zeros((GH, GW), dtype=bool)
        for iy in range(y // CELL, (y + h) // CELL):
            for ix in range(x // CELL, (x + w) // CELL):
                vis[iy, ix] = True
                q.append((ix, iy))
        while q:
            ix, iy = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < GW and 0 <= ny < GH and WALK10[ny, nx] and not vis[ny, nx]:
                    v = ROOM10[ny, nx]
                    vis[ny, nx] = True
                    if v >= 0 and v != r:
                        adj.add(tuple(sorted((r, int(v)))))
                    elif v < 0:
                        q.append((nx, ny))
    return adj


EDGE_SET = {tuple(sorted(e)) for e in EDGES}
