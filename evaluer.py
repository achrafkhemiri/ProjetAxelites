"""Évalue le classifieur sur des tickets qu'il n'a jamais vus."""
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from classifieur import Classifieur, charger


def dedoublonner(df):
    """Le CSV mélange deux exports (ancien et nouvel outil) : un même ticket y figure plusieurs fois.
    On garde un exemplaire de chaque couple (texte, catégorie)."""
    return df.drop_duplicates(subset=['texte', 'categorie']).reset_index(drop=True)


def main():
    brut = charger()
    df = dedoublonner(brut)
    print(f"Tickets : {len(brut)} lignes, {len(df)} après suppression des doublons ({len(brut) - len(df)} copies retirées)")

    # Split groupé par texte : deux tickets identiques (même s'ils ont des étiquettes différentes)
    # ne peuvent pas se retrouver l'un en entraînement et l'autre en test.
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    idx_train, idx_test = next(split.split(df, groups=df['texte']))
    train, test = df.iloc[idx_train], df.iloc[idx_test]
    assert not set(train['texte']) & set(test['texte']), "fuite : un texte est à la fois en train et en test"

    clf = Classifieur().entrainer(train['texte'], train['categorie'])
    acc = accuracy_score(test['categorie'], clf.predire(test['texte']))
    print(f"Entraînement {len(train)}, test {len(test)}")
    print(f"Exactitude (accuracy) sur le jeu de test : {acc:.1%}")


if __name__ == '__main__':
    main()