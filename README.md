# Among Us simplifié — IA multi-agents MAPPO (temps réel)

Jeu inspiré d'Among Us (designs originaux, rendu 2.5D, déplacement libre en temps réel) où **6 crewmates IA et 1 imposteur IA**
sont entraînés par **MAPPO** (Multi-Agent PPO, entraînement centralisé / exécution décentralisée). Tu peux aussi jouer toi-même.

> Projet de fan, non commercial, sans lien avec Innersloth. « Among Us » est une marque d'Innersloth LLC ; ce dépôt
> n'en reprend aucun élément graphique ni sonore.

## Installation
```bash
pip install -r requirements.txt      # numpy + pygame suffisent pour JOUER ; torch + matplotlib pour ENTRAÎNER
```

## Jouer
```bash
python play.py                       # tu es crewmate (modèle : models/mappo.npz)
python play.py --role imposter       # tu es l'imposteur
python play.py --spectate            # 7 IA, tu regardes (Tab = suivre un joueur, +/- = vitesse, P = pause)
python play.py --crew-ai scripted    # change l'IA : mappo | scripted | random   (--imp-ai : + prudent)
python play.py --greedy              # IA déterministes (par défaut elles tirent leur action dans leur politique)
```
Contrôles : `ZQSD`/flèches bouger · `E`/Espace (maintenu) faire une tâche · `K` tuer · `R` signaler un corps (l'imposteur aussi) · `M` carte · `V` vue complète (debug) · `Échap`.
Les réunions (vote) ont leur écran dédié : cartes des joueurs, témoignages, alibis, dépouillement animé, éjection.

## Entraîner (sur ton PC)
```bash
python train_mappo.py --iters 1500 --games 128 --out runs/main
```
- **Reprise** après interruption : relance la même commande avec `--resume` (état complet dans `runs/main/latest.pt`).
- `--workers N` : processus de collecte de parties (défaut : cœurs − 1). **C'est le goulot d'étranglement** (le simulateur
  tourne sur CPU) : plus tu as de cœurs, plus ça va vite. Essaie `--games 256` avec 8+ cœurs.
- `--device auto` : la mise à jour PPO utilise le GPU CUDA s'il est disponible (RTX 2060 : oui, installe la version CUDA de
  PyTorch). Gain modeste, la mise à jour ne représente que ~20 % du temps d'une itération.
- Autres options : `--hidden 128` (taille des réseaux), `--lr`, `--lr-end`, `--ent`, `--ent-end`, `--eval-every`, `--snap-every`.
- Sorties dans `runs/main/` : `latest.npz` (acteurs, c'est ce que lit le jeu), `ckpt_XXXX.npz` tous les 25 itérations,
  `log.csv` (courbes). Pour jouer avec le résultat : `cp runs/main/latest.npz models/mappo.npz`.
- Le modèle fourni dans `models/mappo.npz` contient les réseaux de déplacement et de vote (voir Résultats).
```bash
python plot_training.py runs/main        # results/training_curves.png
python evaluate_mappo.py --n 200         # results/evaluation.md (matrice équipes × imposteurs, IC 95 %)
python -m unittest discover -s tests     # 35 tests
```

### Ajouter le vote appris à un modèle déjà entraîné
```bash
python train_mappo.py --iters 600 --games 128 --out runs/vote --init runs/main/latest.pt --vote-pretrain 2000 --vote-warmup 20
```
- `--vote-pretrain N` : joue N parties avec le vote heuristique et apprend au réseau de vote des crewmates, en
  supervisé, à désigner l'imposteur. **Indispensable** : parti de zéro, le vote apprend à toujours passer (voter au
  hasard coûte cher, donc il renonce avant d'avoir appris à repérer l'imposteur). PPO apprend ensuite quand voter.
- `--init` reprend les réseaux de déplacement d'un autre run et repart de l'itération 0 dans un nouveau dossier.
- `--vote-warmup N` : pendant N itérations seuls les réseaux de vote apprennent (un vote encore aléatoire éjecte des
  innocents ; sans échauffement il dégraderait d'abord les réseaux de déplacement).
- `--vote scripted` : revient à l'ancien comportement (vote heuristique, pas de réseau de vote).
- Un modèle `.npz` sans réseau de vote reste utilisable partout : le jeu retombe sur la règle heuristique.

## Comment ça marche
**Moteur (inchangé)** : vaisseau continu (12 salles + couloirs, collisions, vision par raycast), tâches, kill avec portée et
recharge, témoins, signalement (les crewmates IA signalent dès qu'ils voient un corps ; l'imposteur peut signaler
lui aussi, par choix, comme dans le vrai jeu — un modèle entraîné avant cet ajout ne le fait jamais), réunion avec vote, victoire des crewmates (tâches finies / imposteur éjecté) ou de l'imposteur
(≤ 3 crewmates en vie / temps écoulé). Les témoignages et alibis sont produits par le moteur (du code) ; le choix du
vote est soit une règle heuristique (`env.py`, défaut), soit appris (voir ci-dessous).

**IA : MAPPO hiérarchique**
- *Niveau haut (appris)* : chaque joueur choisit une macro-action quand la précédente est finie — aller dans une salle voisine
  (12), attendre, faire sa tâche, tuer, signaler un corps (16 actions, masquées si illégales). *Niveau bas* : navigation (Dijkstra + collisions).
  Les décisions sont asynchrones, de durée variable → GAE semi-markovien (escompte `γ^(durée/3,5 s)`).
- *Acteur* (décentralisé, `amongus/ai/obs.py`) : observation PARTIELLE ≈ 100 valeurs — position, salle, tâches restantes,
  indice vers la prochaine tâche, ce que je vois maintenant (positions relatives), ma MÉMOIRE (dernière position vue de
  chaque joueur et il y a combien de temps), innocentés, tueurs que j'ai vus, corps visibles ; imposteur : recharge, nombre de
  témoins qui me voient, victime à portée. MLP 2×128 tanh. Un acteur **partagé** par les 6 crewmates, un acteur imposteur.
- *Critique* (centralisé, seulement à l'entraînement) : observation locale + état global (positions, rôles, tâches de tous).
- *Récompenses* (`RewardConfig`) : crew — tâche finie partagée par toute l'équipe, travail, mort ; imposteur — kill (bonus
  s'il n'y a pas de témoin), éjection à tort ; victoire / défaite ±10 ; léger coût de temps.
- *PPO* : clip 0,2, GAE λ = 0,95, 4 époques, entropie 0,02 → 0,003, normalisation des retours, arrêt anticipé sur KL.
- *Auto-jeu avec ligue* : 55 % des parties opposent les derniers réseaux ; sinon l'un des deux camps affronte un ancien snapshot
  ou un bot scripté (imposteur scripté / prudent / aléatoire, crew scripté / aléatoire) — évite que les deux IA se
  spécialisent l'une contre l'autre. La ligue garde les 8 derniers snapshots plus un ancien sur cinq (24 au total,
  `--league-size`), pour ne pas oublier comment battre les adversaires d'il y a longtemps.
- *Vote appris* (`amongus/ai/vote.py`, optionnel) : un second réseau par rôle décide, à chaque réunion, de voter contre
  l'un des 6 autres ou de passer. Il voit ce qui est public (innocentés, accusation d'un témoin direct, qui a signalé,
  et pour chaque joueur la dernière fois qu'un crewmate l'a vu et à quelle distance du corps) plus sa propre mémoire.
  Il sait aussi qui a été « vu avec la victime », et qui l'a été en dernier : c'est ce qui permet de douter de celui
  qui signale le corps.
  Même PPO, même critique centralisé. Chaque vote est un épisode à lui seul, récompensé par le résultat de la réunion
  (imposteur éjecté / innocent éjecté) et, pour les crewmates, par un signal individuel (+1 vote contre l'imposteur,
  −0,5 contre un innocent) : sans lui, des votes dispersés n'éjectent personne et tous apprennent à passer.
- Inférence en **numpy pur** (`amongus/ai/policy.py`) : le jeu n'a pas besoin de PyTorch.

## Résultats du modèle fourni (500 parties/case, % de victoires des crewmates, IC 95 %)
Le modèle de `models/mappo.npz` est entraîné en trois étapes : 1500 itérations sans vote appris (192 000 parties),
600 avec le vote appris et l'imposteur pouvant signaler (76 800 parties), puis 600 de plus après avoir donné au vote
l'information « vu avec la victime » (76 800 parties).

| Équipe \ Imposteur | Imp MAPPO | Imp scripté | Imp prudent | Imp aléatoire |
|---|---|---|---|---|
| Crew MAPPO | 46% ±4 | 96% ±2 | 88% ±3 | 99% ±1 |
| Crew scripté | 41% ±4 | 86% ±3 | 71% ±4 | 98% ±1 |
| Crew aléatoire | 13% ±3 | 53% ±4 | 36% ±4 | 42% ±4 |

Lecture honnête :
- **Le vote appris est ce qui fait la différence pour les crewmates.** Avec le vote heuristique (itération 1500), ils
  faisaient jeu égal avec l'équipe scriptée contre les imposteurs scriptés (87 % / 76 %) ; avec le vote appris, ils
  montent à 96 % / 88 %. Voir `results/eval_iter1500/`.
- **Les deux camps progressent ensemble, d'où les 46 % contre l'imposteur MAPPO.** Ce chiffre a baissé (68 % à
  l'étape 2) parce que l'imposteur est devenu bien plus fort, pas parce que les crewmates ont régressé. Matchs croisés
  entre l'étape 2 (« ancien ») et ce modèle (« nouveau »), 800 parties chacun :

  | Victoires des crewmates | Ancien imposteur | Nouvel imposteur |
  |---|---|---|
  | Anciens crewmates | 62 % | 46 % |
  | Nouveaux crewmates | 72 % | 46 % |

- **Le bluff de l'imposteur marche moins.** L'imposteur signale lui-même ses victimes (15 à 27 % des réunions selon
  l'adversaire). Dans ces réunions, un innocent était éjecté 49 % du temps avec les anciens crewmates ; c'est 35 % avec
  les nouveaux, et 26 % face au nouvel imposteur. Mais celui-ci a déplacé son avantage : quand c'est un crewmate qui
  signale, il n'est plus éjecté que dans 55 % des réunions (79 à 83 % auparavant), et la méfiance a un coût (30 %
  d'innocents éjectés dans ces réunions).
- **Limites.** Les témoignages restent honnêtes et produits par le moteur : l'imposteur ne peut pas mentir, seulement
  choisir où aller, quand tuer, s'il signale, et pour qui voter.

Courbes : `results/training_curves.png` (étape 1), `results/training_curves_vote.png` (étape 2) et
`results/training_curves_vote2.png` (étape 3).

## Structure
`amongus/rt_world.py` monde temps réel + mémoire + enregistrement des trajectoires · `rt_map.py` carte/navigation/vision ·
`rt_gui.py`, `ui.py`, `scene.py` interface · `env.py` règles · `ai/` (obs, politique numpy, MAPPO, bots, rollouts) ·
`train_mappo.py`, `evaluate_mappo.py`, `plot_training.py`, `play.py`, `tests/`.
