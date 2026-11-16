"""Carte du vaisseau : 12 salles reliées par des couloirs (graphe non orienté).

Les distances et le « prochain pas vers X » sont précalculés (BFS) : ils servent
à fabriquer l'état compact des agents (indice de direction vers la tâche la plus proche).
"""
from __future__ import annotations

from collections import deque

ROOMS = [
    "Cafétéria",     # 0  (point de départ et lieu des réunions)
    "Armes",         # 1
    "Navigation",    # 2
    "O2",            # 3
    "Boucliers",     # 4
    "Stockage",      # 5
    "Admin",         # 6
    "Électricité",   # 7
    "Moteurs",       # 8
    "Réacteur",      # 9
    "Sécurité",      # 10
    "Infirmerie",    # 11
]
N_ROOMS = len(ROOMS)

EDGES = [
    (0, 1), (0, 11), (0, 6), (0, 5),
    (1, 3), (1, 2), (3, 2), (3, 4), (2, 4),
    (4, 5), (5, 6), (5, 7),
    (7, 8), (8, 9), (8, 10), (9, 10), (10, 11),
]

# Position des salles pour l'affichage (coordonnées normalisées 0..1)
LAYOUT = {
    0: (0.42, 0.20), 1: (0.66, 0.06), 2: (1.00, 0.20), 3: (0.68, 0.36),
    4: (1.00, 0.64), 5: (0.56, 0.84), 6: (0.66, 0.60), 7: (0.28, 0.86),
    8: (0.00, 0.70), 9: (0.00, 0.36), 10: (0.30, 0.50), 11: (0.16, 0.08),
}

_adj = [[] for _ in range(N_ROOMS)]
for _a, _b in EDGES:
    _adj[_a].append(_b)
    _adj[_b].append(_a)
ADJ = [tuple(sorted(x)) for x in _adj]


def _bfs(src: int) -> list[int]:
    dist = [-1] * N_ROOMS
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in ADJ[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


DIST = [_bfs(s) for s in range(N_ROOMS)]
# NEXT_HOP[a][b] : voisin de a qui rapproche le plus de b (a si a == b)
NEXT_HOP = [
    [a if a == b else min(ADJ[a], key=lambda nb: (DIST[nb][b], nb)) for b in range(N_ROOMS)]
    for a in range(N_ROOMS)
]
