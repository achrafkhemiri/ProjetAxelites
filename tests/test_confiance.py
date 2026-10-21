import numpy as np
import pandas as pd
import pytest

from classifieur import Classifieur, charger, dedoublonner, CATEGORIES
from predire import decider, predire, A_TRIER

CLASSES = ['facturation', 'livraison', 'technique']


@pytest.fixture(scope='module')
def clf():
    df = dedoublonner(charger())
    return Classifieur().entrainer(df['texte'], df['categorie'])


def test_route_si_confiance_suffisante():
    assert decider(np.array([0.10, 0.85, 0.05]), CLASSES, seuil=0.8) == ('livraison', 0.85)


def test_a_trier_si_le_modele_doute():
    categorie, confiance = decider(np.array([0.40, 0.35, 0.25]), CLASSES, seuil=0.8)
    assert categorie == A_TRIER
    assert confiance == pytest.approx(0.40)


def test_seuil_tout_juste_atteint_est_route():
    assert decider(np.array([0.2, 0.8, 0.0]), CLASSES, seuil=0.8)[0] == 'livraison'


def test_predire_renvoie_categorie_et_confiance(clf):
    for categorie, confiance in predire(clf, ["Mon colis n'est jamais arrivé", "Bonjour"]):
        assert categorie in CATEGORIES + [A_TRIER]
        assert 0.0 <= confiance <= 1.0


def test_le_seuil_pilote_le_routage(clf):
    textes = ["Mon colis n'est jamais arrivé", "Le prix affiché n'est pas celui facturé"]
    assert all(c != A_TRIER for c, _ in predire(clf, textes, seuil=0.0))
    assert all(c == A_TRIER for c, _ in predire(clf, textes, seuil=1.01))


def test_dedoublonner_garde_un_exemplaire_par_couple_texte_categorie():
    df = pd.DataFrame({'texte': ['a', 'a', 'a', 'b'], 'categorie': ['compte', 'compte', 'technique', 'compte']})
    assert len(dedoublonner(df)) == 3