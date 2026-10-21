"""Évalue le classifieur et imprime le rapport utilisé dans la fiche de mise en production."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from classifieur import Classifieur, charger


def main():
    df = charger()
    clf = Classifieur()
    train, test = train_test_split(df, test_size=0.25, random_state=0)
    clf.entrainer(train['texte'], train['categorie'])
    pred = clf.predire(test['texte'])
    acc = accuracy_score(test['categorie'], pred)
    print(f"Tickets : {len(df)} (entraînement {len(train)}, test {len(test)})")
    print(f"Exactitude (accuracy) sur le jeu de test : {acc:.1%}")
    print("Verdict :", "OK pour la production" if acc >= 0.9 else "insuffisant")


if __name__ == '__main__':
    main()
