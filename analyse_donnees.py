"""Diagnostic des données : doublons du double export et étiquettes incohérentes.

    python analyse_donnees.py

Les tickets sont des formulations courtes entourées de formules de politesse
(« Bonjour, ... Merci. »). On retire ces formules et les numéros de commande pour
retrouver la demande elle-même, puis on regarde si les humains l'ont toujours
étiquetée de la même façon.
"""
import re

from classifieur import charger
from evaluer import dedoublonner

SALUTATIONS = r"^(Bonjour|Bonsoir|Hello|Salut|Madame, Monsieur)[,.!]?\s*"
FORMULES_FIN = r"\s*(Merci d'avance|Merci|Cordialement|Bonne journée|C'est urgent)\.?$"


def demande(texte):
    """Le cœur du ticket : sans politesse, sans numéro, en minuscules."""
    t = re.sub(SALUTATIONS, "", texte)
    t = re.sub(FORMULES_FIN, "", t)
    t = re.sub(r"\d+", "N", t)
    return t.strip().lower()


def main():
    brut = charger()
    df = dedoublonner(brut)
    print(f"Lignes : {len(brut)} | textes distincts : {brut['texte'].nunique()} "
          f"| copies du double export : {len(brut) - brut['texte'].nunique()}")

    df = df.assign(demande=df['texte'].map(demande))
    repartition = df.groupby('demande')['categorie'].value_counts().unstack(fill_value=0)
    incoherentes = repartition[(repartition > 0).sum(axis=1) > 1]
    n_tickets = incoherentes.to_numpy().sum()
    print(f"Demandes distinctes : {len(repartition)} | étiquetées de façon incohérente : {len(incoherentes)} "
          f"({n_tickets} tickets sur {len(df)}, soit {n_tickets / len(df):.0%})")

    # Plafond : même un modèle parfait ne peut donner qu'UNE catégorie par demande.
    plafond = repartition.max(axis=1).sum() / len(df)
    print(f"Exactitude maximale atteignable avec ces étiquettes : {plafond:.1%}\n")

    print("Demandes étiquetées de façon incohérente :")
    print(incoherentes.loc[incoherentes.sum(axis=1).sort_values(ascending=False).index].to_string())


if __name__ == '__main__':
    main()