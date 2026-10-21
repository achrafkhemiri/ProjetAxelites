"""Classe un ticket et indique la confiance du modèle.

    python predire.py "Mon colis n'est jamais arrivé"
    python predire.py "Mon colis n'est jamais arrivé" --seuil 0.7

Si la confiance est inférieure au seuil, le ticket reste dans la file manuelle (« a_trier »).
"""
import argparse

from classifieur import Classifieur, charger, dedoublonner

# Choisi à partir des données avec `python seuils.py` (voir REPONSES.md, question 3).
SEUIL_CONFIANCE = 0.80
A_TRIER = 'a_trier'


def decider(probas, classes, seuil=SEUIL_CONFIANCE):
    """À partir des probabilités d'un ticket : (catégorie, confiance), ou (a_trier, confiance) si le modèle doute.
    La confiance est la probabilité de la catégorie la plus probable."""
    i = probas.argmax()
    confiance = float(probas[i])
    return (classes[i] if confiance >= seuil else A_TRIER), confiance


def predire(clf, textes, seuil=SEUIL_CONFIANCE):
    """Liste de (catégorie ou a_trier, confiance), un par texte."""
    probas, classes = clf.probabilites(textes)
    return [decider(p, classes, seuil) for p in probas]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('texte', help="texte du ticket")
    parser.add_argument('--seuil', type=float, default=SEUIL_CONFIANCE, help=f"seuil de confiance (défaut {SEUIL_CONFIANCE})")
    args = parser.parse_args()

    df = dedoublonner(charger())
    clf = Classifieur().entrainer(df['texte'], df['categorie'])
    categorie, confiance = predire(clf, [args.texte], args.seuil)[0]
    print(f"{categorie} (confiance {confiance:.0%})")


if __name__ == '__main__':
    main()