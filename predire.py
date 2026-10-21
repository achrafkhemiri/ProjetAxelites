"""Usage : python predire.py "texte du ticket" """
import sys
from classifieur import Classifieur, charger


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    df = charger()
    clf = Classifieur().entrainer(df['texte'], df['categorie'])
    print(clf.predire([sys.argv[1]])[0])


if __name__ == '__main__':
    main()
