"""Classifieur de tickets de support — Maison Lefort.
Catégories : facturation, livraison, technique, compte, produit.
Pipeline : TF-IDF (mots + bigrammes) + régression logistique.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CATEGORIES = ['facturation', 'livraison', 'technique', 'compte', 'produit']


def charger(chemin='data/tickets.csv'):
    df = pd.read_csv(chemin)
    df['texte'] = df['texte'].fillna('').str.strip()
    return df


class Classifieur:
    def __init__(self):
        self.vectoriseur = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, strip_accents='unicode')
        self.modele = LogisticRegression(max_iter=1000, C=5.0)

    def entrainer(self, textes, categories):
        """Ajuste le TF-IDF ET le modèle sur les seuls textes d'entraînement.
        Ajuster le vocabulaire sur des textes de test serait une fuite de données."""
        X = self.vectoriseur.fit_transform(textes)
        self.modele.fit(X, categories)
        return self

    def predire(self, textes):
        X = self.vectoriseur.transform(textes)
        return self.modele.predict(X)

    def probabilites(self, textes):
        X = self.vectoriseur.transform(textes)
        return self.modele.predict_proba(X), list(self.modele.classes_)
