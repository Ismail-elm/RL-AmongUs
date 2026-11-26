"""Génère les captures d'écran du README (sans fenêtre) dans docs/ :

    python tools_screenshots.py [dossier] [modèle.npz]

Un « joueur » scripté fait ses tâches (crewmate) ou chasse (imposteur) pour obtenir des scènes représentatives.
"""
import math
import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pygame

SIZE = (1600, 987)                       # 1200 x 740 logiques agrandis de 4/3, rendu natif
pygame.display.get_desktop_sizes = lambda: [SIZE]

from amongus import rt_map as M
from amongus.ai.claim import ClaimCtrl
from amongus.ai.policy import PolicyCtrl, load_actors, load_claim_actor, load_vote_actors
from amongus.ai.vote import VoteCtrl
from amongus.config import Config
from amongus.rt_gui import RTApp


class Done(Exception):
    pass


def make_app(model, role, seed, human=0):
    ca, ia = load_actors(model)
    cv, iv = load_vote_actors(model)
    ic = load_claim_actor(model)
    return RTApp(Config.players("5+2"), PolicyCtrl(ca, seed=seed), PolicyCtrl(ia, seed=seed + 1), human=human,
                 human_role=role, seed=seed, fullscreen=True, crew_vote=VoteCtrl(cv, seed=seed + 2),
                 imp_vote=VoteCtrl(iv, seed=seed + 3), imp_claim=ClaimCtrl(ic, seed=seed + 4))


def drive(app):
    """Entrée du joueur scripté (comme ZQSD + E)."""
    w = app.world

    def inp():
        h = w.human
        if h is None or not w.alive[h]:
            return dict(dx=0.0, dy=0.0, use=False)
        x, y = w.x[h], w.y[h]
        if h in w.imps:
            crew = w.crew_alive()
            tgt = min(crew, key=lambda q: math.hypot(w.x[q] - x, w.y[q] - y))
            if w.room[tgt] == w.room[h] and w.room[h] >= 0:
                return dict(dx=w.x[tgt] - x, dy=w.y[tgt] - y, use=False)
            r = w.room[tgt] if w.room[tgt] >= 0 else M.nearest_room(w.x[tgt], w.y[tgt])
            dx, dy = M.steer_goal(("room", r), x, y)
            return dict(dx=dx, dy=dy, use=False)
        if not w.tasks[h]:
            return dict(dx=0.0, dy=0.0, use=False)
        r = min(w.tasks[h], key=lambda t: math.hypot(M.task_stand(t)[0] - x, M.task_stand(t)[1] - y))
        sx, sy = M.task_stand(r)
        if math.hypot(sx - x, sy - y) < 30:
            return dict(dx=0.0, dy=0.0, use=True)
        dx, dy = M.steer_goal(("spot", r), x, y)
        return dict(dx=dx, dy=dy, use=False)

    app.read_input = inp


def run_scene(app, want, out):
    """want : {nom_écran: (fichier, délai_s, condition)} ; capture la 1re image qui satisfait chaque entrée."""
    w, done, t_in = app.world, set(), {}
    orig = pygame.display.flip

    def key(k):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))

    def flip():
        fn = sys._getframe(1).f_code.co_name
        now = time.time()
        if fn not in t_in:
            t_in.clear()
            t_in[fn] = now
        since = now - t_in[fn]
        for name, (fname, delay, cond) in want.items():
            if name == fn and fname not in done and since >= delay and cond(app):
                pygame.image.save(pygame.display.get_surface(), os.path.join(out, fname))
                done.add(fname)
                print("  ->", fname, flush=True)
        if len(done) == len(want):
            raise Done
        if fn in ("vote_screen", "human_claim") and since > 2.5:
            if fn == "human_claim":
                pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(170, 190), button=1, logical=1))
            else:
                key(pygame.K_s)
            key(pygame.K_RETURN)
            t_in[fn] = now + 99
        if fn == "show_meeting_result" and since > 3.0:
            key(pygame.K_SPACE)
        if fn == "end_screen" and since > 1.0:
            key(pygame.K_r)
        if fn == "draw" and w.human is not None and w.alive[w.human]:
            if w.human in w.imps and w.human_can_kill() and "kill_ok" in want:
                key(pygame.K_k)
            if w.human_can_report():
                key(pygame.K_r)
        return orig()

    pygame.display.flip = flip
    try:
        app.run()
    except Done:
        pass
    finally:
        pygame.display.flip = orig


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "docs")
    model = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "models", "mappo.npz")
    os.makedirs(out, exist_ok=True)
    alive = lambda a: a.world.human is not None and a.world.alive[a.world.human]
    # 1) menu
    app = make_app(model, "crew", 3)
    orig = pygame.display.flip
    n = [0]

    def menu_flip():
        n[0] += 1
        if n[0] == 20:
            pygame.image.save(pygame.display.get_surface(), os.path.join(out, "menu.png"))
            raise Done
        return orig()

    pygame.display.flip = menu_flip
    try:
        app.main_menu()
    except Done:
        print("  -> menu.png")
    pygame.display.flip = orig
    # 2) crewmate : en jeu, réunion, éjection
    app = make_app(model, "crew", 11)
    drive(app)
    run_scene(app, {
        "draw": ("crewmate.png", 0.0, lambda a: alive(a) and 14 < a.world.time < 40 and a.world.moving[a.world.human]),
        "vote_screen": ("reunion.png", 1.0, lambda a: True),
        "eject_animation": ("ejection.png", 1.9, lambda a: True),
    }, out)
    # 3) imposteur : boutons d'action
    app = make_app(model, "imposter", 5)
    drive(app)
    run_scene(app, {"draw": ("imposteur.png", 0.0, lambda a: alive(a) and a.world.time > 12
                             and any(a.world.room[q] == a.world.room[a.world.human] for q in a.world.crew_alive()))},
              out)
    # 4) spectateur
    app = make_app(model, "crew", 7, human=None)
    app.free_cam = False
    run_scene(app, {"draw": ("spectateur.png", 0.0, lambda a: a.world.time > 25)}, out)


if __name__ == "__main__":
    main()
