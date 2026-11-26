"""Affichage sans fenêtre (pilote vidéo « dummy ») : rendu à la résolution de l'écran, menu, caméra libre, boutons."""
import os
import sys
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from amongus import canvas as D
from amongus.ai.bots import ScriptedCrew, ScriptedImposter
from amongus.config import Config


def app(human=0, role="crew", size=(1920, 1080), fullscreen=True, menu=False):
    from amongus.rt_gui import RTApp
    pygame.display.get_desktop_sizes = lambda: [size]
    a = RTApp(Config.players("5+2"), ScriptedCrew(), ScriptedImposter(1), human=human, human_role=role,
              fullscreen=fullscreen, menu=menu, seed=1)
    a.anim_scale = 0
    return a


class TestCanvas(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def test_conversions(self):
        real = pygame.Surface((2400, 1480))
        c = D.Canvas(real, 2.0, (0, 0), (1200, 740))
        self.assertEqual(c.R((10, 20, 30, 40)), pygame.Rect(20, 40, 60, 80))
        self.assertEqual(c.to_logical((200, 100)), (100, 50))
        self.assertEqual(c.L(3), 6)
        self.assertEqual(c.get_rect(center=(5, 5)).center, (5, 5))

    def test_shapes_drawn_at_full_resolution(self):
        real = pygame.Surface((400, 400))
        c = D.Canvas(real, 4.0, (0, 0), (100, 100))
        D.rect(c, (255, 0, 0), (10, 10, 10, 10))
        self.assertEqual(real.get_at((45, 45))[:3], (255, 0, 0))      # 10..20 logique -> 40..80 réel
        self.assertEqual(real.get_at((85, 85))[:3], (0, 0, 0))
        plain = pygame.Surface((50, 50))
        D.circle(plain, (0, 255, 0), (25, 25), 5)                        # surface ordinaire : pygame tel quel
        self.assertEqual(plain.get_at((25, 25))[:3], (0, 255, 0))

    def test_text_is_rendered_large(self):
        real = pygame.Surface((2400, 1480))
        c = D.Canvas(real, 2.0, (0, 0), (1200, 740))
        f = D.font(16, True)
        r = D.label(c, f, "Bonjour", outlined=False, topleft=(10, 10))
        w1 = f.size("Bonjour")[0]
        self.assertAlmostEqual(r.w, w1, delta=3)                          # rectangle LOGIQUE renvoyé
        self.assertGreater(D.hifont(f, 2.0).size("Bonjour")[0], 1.8 * w1)  # police réelle deux fois plus grande


class TestApp(unittest.TestCase):
    def test_native_resolution_and_toggle(self):
        a = app()
        self.assertEqual(a.screen.real.get_size(), (1920, 1080))
        self.assertGreater(a.screen.k, 1.4)
        a.toggle_fullscreen()
        self.assertFalse(a.fullscreen)
        self.assertLess(a.screen.real.get_width(), 1920)
        a.toggle_fullscreen()
        self.assertEqual(a.screen.real.get_size(), (1920, 1080))
        a.draw()

    def test_mouse_events_are_logical(self):
        a = app()
        k, ox, oy = a.screen.k, a.screen.ox, a.screen.oy
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(round(ox + 600 * k), round(oy + 370 * k)),
                                             button=1))
        evs = [e for e in a.poll() if e.type == pygame.MOUSEBUTTONDOWN]
        self.assertEqual(len(evs), 1)
        self.assertLessEqual(abs(evs[0].pos[0] - 600) + abs(evs[0].pos[1] - 370), 2)

    def test_menu_choices(self):
        a = app(menu=True)
        for key, mode in ((pygame.K_1, "crew"), (pygame.K_2, "imposter"), (pygame.K_3, "spectate")):
            pygame.event.clear()
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))
            self.assertEqual(a.main_menu(), mode)
        a.set_mode("spectate")
        self.assertIsNone(a.human)
        self.assertTrue(a.free_cam)

    def test_free_camera(self):
        a = app(human=None)
        a.free_cam = True
        a.center_camera(2000, 1000)
        cam = list(a.cam)
        a.update_camera(0.1)                     # caméra libre : elle ne revient pas sur un joueur
        self.assertEqual(a.cam, cam)
        a.center_camera(-5000, -5000)            # bornée au vaisseau
        self.assertGreater(a.cam[0], -1200)

    def test_imposter_buttons(self):
        a = app(role="imposter")
        self.assertIn(a.human, a.world.imps)
        rects = [a.btn_use, a.btn_report, a.btn_vent, a.btn_sabotage, a.btn_emergency]
        for i, r in enumerate(rects):            # aucun bouton ne se chevauche
            for q in rects[i + 1:]:
                self.assertFalse(r.colliderect(q), (r, q))
        a.draw()
        crew = app(role="crew")
        self.assertFalse(crew.btn_emergency.colliderect(crew.btn_use))
        crew.draw()


if __name__ == "__main__":
    unittest.main()
