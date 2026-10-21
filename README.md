# Tri automatique des tickets de support — Maison Lefort

Classe chaque ticket entrant dans une des cinq catégories : `facturation`, `livraison`, `technique`, `compte`, `produit`. Le service client veut router automatiquement les tickets vers la bonne équipe. Quand le modèle n'est pas sûr, le ticket reste dans la file manuelle (`a_trier`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluer.py                          # rapport de performance par catégorie, sans et avec seuil
python evaluer.py --test nouveaux.csv      # évaluation sur des tickets inédits (colonnes texte, categorie)
python seuils.py                           # tableau routage / erreur par seuil, choix du seuil
python analyse_donnees.py                  # diagnostic des données : doublons, étiquettes incohérentes
python predire.py "Mon colis n'est jamais arrivé"              # -> livraison (confiance 83%)
python predire.py "Mon colis n'est jamais arrivé" --seuil 0.9  # -> a_trier (confiance 83%)
python -m pytest tests
```

## Données

`data/tickets.csv` : 770 lignes exportées de l'ancien outil de ticketing et de l'actuel, étiquetées à la main par le service client. Colonnes `id`, `texte`, `categorie`.

- Le double export crée des doublons : il n'y a que 445 textes distincts. Ils sont retirés avant l'entraînement et l'évaluation (`dedoublonner`), ce qui laisse 448 tickets.
- 20 demandes, soit 36 % des tickets, sont étiquetées de façon incohérente (exemple : « Le prix affiché n'est pas celui facturé » est classé facturation, produit ou technique). Un guide d'annotation est nécessaire ; voir `analyse_donnees.py`.

## Modèle

TF-IDF (mots et bigrammes) et régression logistique. Le TF-IDF est ajusté uniquement sur les données d'entraînement.

## Performance réelle

L'ancienne fiche de mise en production annonçait 91 % : ce chiffre était gonflé par les doublons entre l'entraînement et le test (voir `REPONSES.md`).

Validation croisée groupée par texte (5 plis × 5 répétitions), sans doublons :

- **sans seuil** : 80,3 % d'exactitude, et aucune catégorie ne respecte la règle métier (erreur de 14 à 26 % selon la file) ;
- **avec le seuil de confiance 0,80** : 54 % des tickets sont routés automatiquement avec 2,1 % d'erreur, et toutes les files sont sous 4 % ; 46 % restent en manuel.

## Règle métier

Le routage automatique n'est acceptable que si le modèle se trompe sur **moins d'un ticket sur dix** par catégorie. En dessous, les tickets doivent rester dans la file manuelle.

Le seuil (`SEUIL_CONFIANCE` dans `predire.py`) est le plus petit qui respecte cette règle dans chaque file, même dans le pire tirage de la validation croisée (`python seuils.py`).