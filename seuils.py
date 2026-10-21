"""Choisit le seuil de confiance à partir des données.

Pour chaque seuil : part des tickets routés automatiquement, et taux d'erreur sur ces tickets
(global, et dans la pire file). Les prédictions viennent de la validation croisée groupée
d'evaluer.py : chaque ticket est prédit par un modèle qui ne l'a jamais vu.

Règle métier (README) : moins d'un ticket sur dix mal routé, PAR catégorie.
Pour garder une marge, on exige que la règle tienne dans CHAQUE répétition de la validation
croisée (le pire tirage), pas seulement en moyenne.
"""
import numpy as np
import pandas as pd

from classifieur import charger, dedoublonner
from evaluer import predictions_hors_echantillon, ERREUR_MAX
from predire import SEUIL_CONFIANCE

SEUILS = [round(s, 2) for s in np.arange(0.40, 0.91, 0.05)]


def tableau(p, seuils=SEUILS):
    p = p.assign(erreur=p['vrai'] != p['pred'])
    lignes = []
    for s in seuils:
        routes = p[p['confiance'] >= s]
        par_file = routes.groupby('pred')['erreur'].mean()
        pire_tirage = routes.groupby(['repetition', 'pred'])['erreur'].mean().max()
        lignes.append({
            'seuil': s,
            'routes_auto': len(routes) / len(p),
            'erreur_routes': routes['erreur'].mean(),
            'pire_file': f"{par_file.idxmax()} {par_file.max():.1%}",
            'pire_file_pire_tirage': pire_tirage,
            'erreurs_evitees': (p['erreur'] & (p['confiance'] < s)).sum() / p['erreur'].sum(),
        })
    return pd.DataFrame(lignes)


def seuil_recommande(t):
    """Plus petit seuil (donc le plus de routage automatique) qui respecte la règle même dans le pire tirage."""
    ok = t[t['pire_file_pire_tirage'] < ERREUR_MAX]
    return float(ok['seuil'].min()) if len(ok) else None


def main():
    p = predictions_hors_echantillon(dedoublonner(charger()))
    t = tableau(p)
    pct = '{:.1%}'.format
    print(t.to_string(index=False, formatters={k: pct for k in
          ['routes_auto', 'erreur_routes', 'pire_file_pire_tirage', 'erreurs_evitees']}))
    s = seuil_recommande(t)
    print(f"\nSeuil recommandé : {s}  |  seuil en production (predire.py) : {SEUIL_CONFIANCE}")


if __name__ == '__main__':
    main()