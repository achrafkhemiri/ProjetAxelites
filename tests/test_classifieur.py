from classifieur import Classifieur, charger, CATEGORIES


def test_categories_connues():
    df = charger()
    assert set(df['categorie'].unique()) == set(CATEGORIES)


def test_predit_une_categorie_connue():
    df = charger()
    clf = Classifieur().preparer(df['texte']).entrainer(df['texte'], df['categorie'])
    assert clf.predire(["Ma commande n'est pas arrivée"])[0] in CATEGORIES
