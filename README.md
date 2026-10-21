# Tri automatique des tickets de support — Maison Lefort

Classe chaque ticket entrant dans une des cinq catégories : `facturation`, `livraison`, `technique`, `compte`, `produit`. Le service client veut router automatiquement les tickets vers la bonne équipe.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluer.py          # rapport de performance
python predire.py "Mon colis n'est jamais arrivé"
python -m pytest tests     # (pip install pytest)
```

## Données

`data/tickets.csv` : 770 tickets réels anonymisés, exportés de l'ancien outil de ticketing et de l'actuel, étiquetés à la main par le service client. Colonnes `id`, `texte`, `categorie`.

## Modèle

TF-IDF (mots et bigrammes) + régression logistique. Le rapport `evaluer.py` donne **91 % d'exactitude** sur le jeu de test, c'est le chiffre qui figure dans la fiche de mise en production.

## Règle métier

Le routage automatique n'est acceptable que si le modèle se trompe sur **moins d'un ticket sur dix** par catégorie. En dessous, les tickets doivent rester dans la file manuelle.
