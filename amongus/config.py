"""Paramètres du jeu et des récompenses (modifiables)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

PLAYER_NAMES = ["Rouge", "Bleu", "Vert", "Rose", "Orange", "Jaune", "Cyan", "Blanc"]


@dataclass
class Config:
    # --- règles ---
    n_crew: int = 5              # crewmates
    n_imp: int = 2               # imposteurs (le jeu tour par tour historique n'en gère qu'un)
    tasks_per_crew: int = 4      # tâches par crewmate (5 avant : la victoire par les tâches était hors de portée)
    task_duration: int = 4       # nb de tours consécutifs dans la salle pour finir une tâche
    kill_cooldown: int = 3       # tours entre deux kills (et après chaque réunion)
    crew_left_to_lose: int = 0   # les imposteurs gagnent quand les crewmates vivants ne sont pas plus nombreux
                                 # qu'eux (vrai jeu) ; on peut exiger moins : gagner dès qu'il en reste <= cette valeur
    max_steps: int = 100         # temps limite : si la partie dure trop, l'imposteur gagne
    skip_threshold: int = 3      # un crewmate n'ose voter que s'il a <= 3 suspects, sinon il "skip"
    recall: float = 0.5          # en réunion, proba qu'un crewmate se souvienne de (et parle de) ce qu'il a vu d'un joueur
    # --- récompenses ---
    r_step: float = -0.01        # petite pénalité par tour (pousse à agir)
    r_task: float = 2.0          # tâche terminée
    r_task_progress: float = 0.25  # chaque tour passé à travailler sur une tâche (aide l'apprentissage)
    r_kill: float = 2.0          # kill réussi (imposteur)
    r_death: float = -2.0        # mort / éjection injuste (crewmate)
    r_crew_win: float = 5.0      # victoire des crewmates (pour ceux encore en vie)
    r_crew_loss: float = -5.0
    r_imp_win: float = 10.0
    r_imp_loss: float = -10.0    # inclut l'éjection de l'imposteur

    @classmethod
    def players(cls, spec: str | None = None) -> "Config":
        """Config pour « 5+2 » (5 crewmates, 2 imposteurs). Sans argument : variable d'environnement AMONG_PLAYERS
        (posée par les scripts pour leurs processus de collecte), sinon les valeurs par défaut."""
        import os
        spec = spec or os.environ.get("AMONG_PLAYERS")
        if not spec:
            return cls()
        crew, imp = (int(v) for v in spec.split("+"))
        return cls(n_crew=crew, n_imp=imp)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
