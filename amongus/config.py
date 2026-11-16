"""Paramètres du jeu et des récompenses (modifiables)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

PLAYER_NAMES = ["Rouge", "Bleu", "Vert", "Rose", "Orange", "Jaune", "Cyan", "Blanc"]


@dataclass
class Config:
    # --- règles ---
    n_crew: int = 6              # 6 crewmates + 1 imposteur = 7 joueurs (toi + 5 IA + imposteur IA)
    tasks_per_crew: int = 5      # tâches par crewmate
    task_duration: int = 4       # nb de tours consécutifs dans la salle pour finir une tâche
    kill_cooldown: int = 3       # tours entre deux kills (et après chaque réunion)
    crew_left_to_lose: int = 3   # l'imposteur gagne s'il reste <= 3 crewmates en vie
    max_steps: int = 100         # temps limite : si la partie dure trop, l'imposteur gagne
    skip_threshold: int = 3      # un crewmate n'ose voter que s'il a <= 3 suspects, sinon il "skip"
    recall: float = 0.5          # proba qu'un crewmate se souvienne d'avoir vu quelqu'un à un instant donné
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
