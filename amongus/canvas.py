"""Rendu net à n'importe quelle résolution.

Tout le code d'affichage raisonne dans un repère logique de W x H pixels (1200 x 740). Un `Canvas` se fait passer
pour la surface d'écran : il convertit chaque tracé (rectangle, cercle, ligne, polygone, texte) vers la vraie
résolution de l'écran au lieu d'agrandir une petite image. Formes et textes sont donc tracés nativement, sans flou.

Les fonctions `rect`, `circle`, `line`, ... remplacent `pygame.draw.*` : sur une surface ordinaire elles se
comportent exactement comme pygame ; sur un `Canvas`, elles tracent à l'échelle.
"""
from __future__ import annotations

import pygame

_FONT_SPEC: dict[int, tuple] = {}          # id(police logique) -> (taille, gras, police)
_HI_FONTS: dict[tuple, pygame.font.Font] = {}
FONT_NAMES = "arialroundedmtbold,trebuchetms,segoeui,dejavusans,arial"


def font(size, bold=False):
    """Police logique (taille en pixels logiques), enregistrée pour pouvoir en tirer une version haute résolution."""
    key = (int(size), bool(bold), 1.0)
    f = _HI_FONTS.get(key)
    if f is None:
        f = _HI_FONTS[key] = pygame.font.SysFont(FONT_NAMES, int(size), bold=bold)
        _FONT_SPEC[id(f)] = (int(size), bool(bold), f)
    return f


def hifont(f, k):
    """Version de la police `f` à l'échelle k (ou f elle-même si inconnue)."""
    spec = _FONT_SPEC.get(id(f))
    if spec is None or spec[2] is not f or abs(k - 1.0) < 1e-6:
        return f
    size, bold, _ = spec
    key = (size, bold, round(k, 3))
    g = _HI_FONTS.get(key)
    if g is None:
        g = _HI_FONTS[key] = pygame.font.SysFont(FONT_NAMES, max(1, round(size * k)), bold=bold)
    return g


class Canvas:
    """Surface logique (w x h) tracée à l'échelle k sur `real`, décalée de `off` (bandes noires éventuelles)."""

    def __init__(self, real: pygame.Surface, k: float, off=(0.0, 0.0), size=None):
        self.real, self.k = real, float(k)
        self.ox, self.oy = off
        self.w, self.h = size if size is not None else (round(real.get_width() / k), round(real.get_height() / k))
        self._hires: dict[int, tuple] = {}
        self._clip = None

    # ------------------------------------------------------------- conversions
    def P(self, x, y):
        return (self.ox + x * self.k, self.oy + y * self.k)

    def R(self, r, offset=True):
        if hasattr(r, "w"):
            x, y, w, h = r.x, r.y, r.w, r.h
        else:
            x, y, w, h = r
        ox, oy = (self.ox, self.oy) if offset else (0.0, 0.0)
        x0, y0 = round(ox + x * self.k), round(oy + y * self.k)
        x1, y1 = round(ox + (x + w) * self.k), round(oy + (y + h) * self.k)
        return pygame.Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def L(self, v):
        """Épaisseur / rayon logique -> pixels réels (au moins 1 si non nul)."""
        return 0 if v == 0 else max(1, round(v * self.k))

    def to_logical(self, pos):
        return (int((pos[0] - self.ox) / self.k), int((pos[1] - self.oy) / self.k))

    # ----------------------------------------------------------- API « Surface »
    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_size(self):
        return (self.w, self.h)

    def get_rect(self, **kw):
        return _rect_with(self.w, self.h, kw)

    def fill(self, color, rect=None, special_flags=0):
        if rect is None:
            return self.real.fill(color, None, special_flags)
        return self.real.fill(color, self.R(rect), special_flags)

    def set_clip(self, r):
        self._clip = None if r is None else pygame.Rect(r)
        self.real.set_clip(None if r is None else self.R(r))

    def get_clip(self):
        return self._clip if self._clip is not None else pygame.Rect(0, 0, self.w, self.h)

    def register(self, src: pygame.Surface, hi: pygame.Surface, hk: float):
        """`hi` est `src` dessinée à l'échelle hk : elle sera utilisée à sa place (net)."""
        self._hires[id(src)] = (src, hi, hk)

    def blit(self, src, dest, area=None, special_flags=0):
        dx, dy = (dest.x, dest.y) if isinstance(dest, pygame.Rect) else (dest[0], dest[1])
        k = self.k
        ent = self._hires.get(id(src))
        if ent is not None and ent[0] is src:
            _, hi, hk = ent
            if area is None:
                area = pygame.Rect(0, 0, src.get_width(), src.get_height())
            if abs(hk - k) < 1e-3:
                return self.real.blit(hi, self.P(dx, dy), self.R(area, offset=False), special_flags)
            a = pygame.Rect(round(area[0] * hk), round(area[1] * hk), round(area[2] * hk), round(area[3] * hk))
            a = a.clip(hi.get_rect())
            if a.w <= 0 or a.h <= 0:
                return None
            piece = hi.subsurface(a)
            dx += (a.x / hk) - area[0]
            dy += (a.y / hk) - area[1]
            out = pygame.transform.smoothscale(piece, (max(1, round(a.w * k / hk)), max(1, round(a.h * k / hk))))
            return self.real.blit(out, self.P(dx, dy), None, special_flags)
        if area is not None:
            a = pygame.Rect(area).clip(src.get_rect())
            if a.w <= 0 or a.h <= 0:
                return None
            dx += a.x - area[0]
            dy += a.y - area[1]
            src = src.subsurface(a)
        return self.blit_stretch(src, (dx, dy, src.get_width(), src.get_height()), special_flags)

    def blit_stretch(self, src, rect, special_flags=0):
        """Étire `src` sur le rectangle logique `rect`."""
        r = self.R(rect)
        if r.w <= 0 or r.h <= 0:
            return None
        if src.get_size() == r.size:
            out = src
        elif src.get_bitsize() >= 24:
            out = pygame.transform.smoothscale(src, r.size)
        else:
            out = pygame.transform.scale(src, r.size)
        return self.real.blit(out, r.topleft, None, special_flags)

    def blit_native(self, img, **anchor):
        """Pose une image déjà à la résolution réelle ; `anchor` en coordonnées logiques (center=..., topleft=...)."""
        r = img.get_rect()
        (name, pos), = anchor.items()
        setattr(r, name, tuple(round(v) for v in self.P(*pos)))
        return self.real.blit(img, r)


def _rect_with(w, h, kw):
    r = pygame.Rect(0, 0, w, h)
    for name, v in kw.items():
        setattr(r, name, v)
    return r


def is_canvas(s):
    return isinstance(s, Canvas)


def sprite(w, h, scale, draw, k):
    """Surface réelle d'un petit dessin logique (w x h) agrandi `scale` fois, tracée nativement à l'échelle k.
    `draw(canvas)` dessine dans le repère logique (w x h)."""
    hk = scale * k
    surf = pygame.Surface((max(1, round(w * hk)), max(1, round(h * hk))), pygame.SRCALPHA)
    draw(Canvas(surf, hk, (0, 0), (w, h)))
    return surf


# ------------------------------------------------------------ tracés (pygame.draw)
def rect(surf, color, r, width=0, border_radius=0, border_top_left_radius=-1, border_top_right_radius=-1,
         border_bottom_left_radius=-1, border_bottom_right_radius=-1):
    if not isinstance(surf, Canvas):
        return pygame.draw.rect(surf, color, r, width, border_radius, border_top_left_radius, border_top_right_radius,
                                border_bottom_left_radius, border_bottom_right_radius)
    s = lambda v: v if v <= 0 else max(1, round(v * surf.k))
    return pygame.draw.rect(surf.real, color, surf.R(r), surf.L(width), s(border_radius), s(border_top_left_radius),
                            s(border_top_right_radius), s(border_bottom_left_radius), s(border_bottom_right_radius))


def circle(surf, color, center, radius, width=0):
    if not isinstance(surf, Canvas):
        return pygame.draw.circle(surf, color, center, radius, width)
    return pygame.draw.circle(surf.real, color, surf.P(*center), max(0.5, radius * surf.k), surf.L(width))


def line(surf, color, start, end, width=1):
    if not isinstance(surf, Canvas):
        return pygame.draw.line(surf, color, start, end, width)
    return pygame.draw.line(surf.real, color, surf.P(*start), surf.P(*end), surf.L(width))


def polygon(surf, color, points, width=0):
    if not isinstance(surf, Canvas):
        return pygame.draw.polygon(surf, color, points, width)
    return pygame.draw.polygon(surf.real, color, [surf.P(*p) for p in points], surf.L(width))


def ellipse(surf, color, r, width=0):
    if not isinstance(surf, Canvas):
        return pygame.draw.ellipse(surf, color, r, width)
    return pygame.draw.ellipse(surf.real, color, surf.R(r), surf.L(width))


def arc(surf, color, r, start, stop, width=1):
    if not isinstance(surf, Canvas):
        return pygame.draw.arc(surf, color, r, start, stop, width)
    return pygame.draw.arc(surf.real, color, surf.R(r), start, stop, surf.L(width))


def label(surf, f, s, color=(255, 255, 255), width=2, outlined=True, **anchor):
    """Texte (cerné de noir si `outlined`) posé par une ancre logique (center=..., midbottom=..., topleft=...).
    Sur un Canvas, il est rendu à la résolution réelle. Renvoie le rectangle LOGIQUE occupé."""
    from . import style as ST
    if isinstance(surf, Canvas):
        k = surf.k
        hf = hifont(f, k)
        img = ST.outlined_text(hf, s, color, max(1, round(width * k))) if outlined else hf.render(s, True, color)
        (name, pos), = anchor.items()
        lr = pygame.Rect(0, 0, round(img.get_width() / k), round(img.get_height() / k))
        setattr(lr, name, pos)
        r = img.get_rect()
        setattr(r, name, tuple(round(v) for v in surf.P(*pos)))
        surf.real.blit(img, r)
        return lr
    img = ST.outlined_text(f, s, color, width) if outlined else f.render(s, True, color)
    r = img.get_rect()
    (name, pos), = anchor.items()
    setattr(r, name, pos)
    surf.blit(img, r)
    return r
