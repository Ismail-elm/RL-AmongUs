"""Style graphique « cartoon spatial » : aplats de couleurs franches, contours noirs épais, sols métalliques,
textes blancs cernés de noir. Palette et petites fonctions de dessin partagées par la carte, les personnages et
l'interface. (Rendu seulement : rien ici n'influence les règles ni l'entraînement.)"""
from __future__ import annotations

import pygame

from . import canvas as D

OUTLINE = (17, 17, 17)
SPACE = (14, 17, 26)
STEEL = (154, 166, 184)          # sol métallique
STEEL_LINE = (127, 139, 160)     # joints des dalles
WALL = (95, 107, 128)            # murs
WALL_DARK = (66, 75, 92)
CORRIDOR = (135, 147, 166)
PANEL = (201, 211, 227)          # panneaux d'interface (réunion, HUD)
CARD = (244, 246, 250)
INK = (22, 26, 36)               # texte foncé sur fond clair
MUTED = (74, 82, 102)
GREEN = (76, 203, 94)
RED = (226, 56, 63)
YELLOW = (255, 210, 63)
CYAN = (143, 227, 255)
BLUE_BG = (26, 36, 64)


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def mix(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def orect(surf, color, rect, radius=8, width=3):
    """Rectangle (arrondi) plein, cerné d'un contour noir de `width` px (le contour déborde à l'extérieur)."""
    rect = pygame.Rect(rect)
    D.rect(surf, OUTLINE, rect.inflate(2 * width, 2 * width), border_radius=radius + width if radius else 0)
    D.rect(surf, color, rect, border_radius=radius)


def ocircle(surf, color, center, r, width=3):
    D.circle(surf, OUTLINE, center, r + width)
    D.circle(surf, color, center, r)


def oellipse(surf, color, rect, width=3):
    rect = pygame.Rect(rect)
    D.ellipse(surf, OUTLINE, rect.inflate(2 * width, 2 * width))
    D.ellipse(surf, color, rect)


def outline_sprite(sprite: pygame.Surface, width=3, threshold=160) -> pygame.Surface:
    """Ajoute un contour noir autour des parties opaques d'un sprite (les halos transparents sont ignorés)."""
    mask = pygame.mask.from_surface(sprite, threshold)
    out = pygame.Surface((sprite.get_width() + 2 * width, sprite.get_height() + 2 * width), pygame.SRCALPHA)
    sil = mask.to_surface(setcolor=(*OUTLINE, 255), unsetcolor=(0, 0, 0, 0))
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx * dx + dy * dy <= width * width + 1:
                out.blit(sil, (width + dx, width + dy))
    out.blit(sprite, (width, width))
    return out


_text_cache: dict = {}


def outlined_text(font, s, color=(255, 255, 255), width=2, outline=OUTLINE) -> pygame.Surface:
    """Texte cerné (mis en cache) : lisible sur n'importe quel fond."""
    key = (id(font), s, color, width, outline)
    surf = _text_cache.get(key)
    if surf is not None:
        return surf
    base = font.render(s, True, color)
    edge = font.render(s, True, outline)
    w, h = base.get_width() + 2 * width, base.get_height() + 2 * width
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx or dy:
                surf.blit(edge, (width + dx, width + dy))
    surf.blit(base, (width, width))
    if len(_text_cache) > 600:
        _text_cache.clear()
    _text_cache[key] = surf
    return surf
