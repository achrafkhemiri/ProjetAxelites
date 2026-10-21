"""Évalue le classifieur sur des tickets qu'il n'a jamais vus, catégorie par catégorie.

    python evaluer.py                        # validation croisée sur data/tickets.csv
    python evaluer.py --test nouveaux.csv    # entraîne sur data/tickets.csv, évalue sur un autre fichier
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

from classifieur import Classifieur, charger, CATEGORIES

ERREUR_MAX = 0.10  # règle métier du README : moins d'un ticket sur dix mal routé, par catégorie


def dedoublonner(df):
    """Le CSV mélange deux exports (ancien et nouvel outil) : un même ticket y figure plusieurs fois.
    On garde un exemplaire de chaque couple (texte, catégorie)."""
    return df.drop_duplicates(subset=['texte', 'categorie']).reset_index(drop=True)


def textes_contradictoires(df):
    """Textes identiques étiquetés dans des catégories différentes (problème d'annotation)."""
    n = df.groupby('texte')['categorie'].nunique()
    return n[n > 1].index.tolist()


def predictions_hors_echantillon(df, n_plis=5, n_repetitions=5):
    """Validation croisée groupée par texte, répétée avec plusieurs tirages.
    Chaque ticket est prédit par un modèle qui ne l'a jamais vu (ni lui, ni un doublon)."""
    lignes = []
    for rep in range(n_repetitions):
        cv = StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=rep)
        for idx_train, idx_test in cv.split(df, df['categorie'], groups=df['texte']):
            train, test = df.iloc[idx_train], df.iloc[idx_test]
            assert not set(train['texte']) & set(test['texte'])
            clf = Classifieur().entrainer(train['texte'], train['categorie'])
            lignes.append(predictions(clf, test).assign(repetition=rep))
    return pd.concat(lignes, ignore_index=True)


def predictions(clf, test):
    probas, classes = clf.probabilites(test['texte'])
    return pd.DataFrame({
        'vrai': test['categorie'].to_numpy(),
        'pred': np.array(classes)[probas.argmax(axis=1)],
        'confiance': probas.max(axis=1),
    })


def rapport(p):
    """Par catégorie : erreur dans la file (ce que reçoit l'équipe) et tickets ratés."""
    n_rep = p['repetition'].nunique() if 'repetition' in p else 1
    lignes = []
    for c in CATEGORIES:
        envoyes = p[p['pred'] == c]
        attendus = p[p['vrai'] == c]
        err_file = (envoyes['vrai'] != c).mean() if len(envoyes) else float('nan')
        lignes.append({
            'categorie': c,
            'tickets': len(attendus) // n_rep,
            'precision': 1 - err_file,
            'rappel': (attendus['pred'] == c).mean(),
            'erreur_dans_la_file': err_file,
            'regle_10%': 'OK' if err_file < ERREUR_MAX else 'NON',
        })
    return pd.DataFrame(lignes).set_index('categorie')


def afficher(p):
    exactitude = (p['vrai'] == p['pred']).mean()
    print(f"Exactitude globale : {exactitude:.1%}  (erreur {1 - exactitude:.1%})")
    if 'repetition' in p and p['repetition'].nunique() > 1:
        par_rep = (p['vrai'] == p['pred']).groupby(p['repetition']).mean()
        print(f"  selon le tirage : de {par_rep.min():.1%} à {par_rep.max():.1%}")
    r = rapport(p)
    print("\nPar catégorie :")
    print(r.to_string(formatters={k: '{:.1%}'.format for k in ['precision', 'rappel', 'erreur_dans_la_file']}))
    print("\nMatrice de confusion (lignes = vraie catégorie, colonnes = prédite, cumul des répétitions) :")
    print(pd.DataFrame(confusion_matrix(p['vrai'], p['pred'], labels=CATEGORIES), index=CATEGORIES, columns=CATEGORIES))
    ok = (r['regle_10%'] == 'OK').all()
    print("\nVerdict :", "routage automatique acceptable" if ok else
          "règle métier NON respectée : " + ", ".join(r.index[r['regle_10%'] != 'OK']))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test', help="CSV de tickets inédits (colonnes texte, categorie)")
    args = parser.parse_args()

    brut = charger()
    df = dedoublonner(brut)
    print(f"Tickets : {len(brut)} lignes, {len(df)} après suppression des doublons ({len(brut) - len(df)} copies retirées)")
    contradictoires = textes_contradictoires(df)
    if contradictoires:
        print(f"Attention : {len(contradictoires)} textes identiques portent des étiquettes différentes")

    if args.test:
        test = charger(args.test)
        deja_vus = test['texte'].isin(df['texte']).sum()
        print(f"Jeu de test externe : {len(test)} tickets ({deja_vus} déjà présents dans les données d'entraînement)\n")
        clf = Classifieur().entrainer(df['texte'], df['categorie'])
        afficher(predictions(clf, test))
    else:
        print("Validation croisée 5 plis x 5 répétitions, groupée par texte\n")
        afficher(predictions_hors_echantillon(df))


if __name__ == '__main__':
    main()