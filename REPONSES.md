# Réponses

Répondez en quelques lignes par question, en français, avec vos mots.

## 1. L'écart

Quelles sont les causes de l'écart entre les 91 % annoncés et la réalité ? Pour chacune : où vous l'avez vue, et combien elle pèse.

> **1. Doublons du double export (cause principale, environ 10 points).** Le README indique que les données viennent de l'ancien outil et de l'actuel : sur 770 lignes, il n'y a que 445 textes distincts (`analyse_donnees.py`). Avec le `train_test_split` d'origine, 65 % des tickets de test avaient une copie exacte en entraînement : on mesurait la mémoire du modèle, pas sa capacité à classer un nouveau ticket. En dédoublonnant et en séparant par texte, on passe de 90,7 % à 82,3 % sur le même type de split, et à 80,3 % en validation croisée.
>
> **2. TF-IDF ajusté sur tout le jeu avant le split (environ 0 point).** `preparer(df['texte'])` dans `evaluer.py`. C'est une vraie fuite de méthode, corrigée (le TF-IDF est maintenant ajusté dans `entrainer`, sur l'entraînement seulement), mais elle ne pèse rien ici : un mot vu seulement dans le test garde un poids nul dans la régression, donc seuls les IDF changent un peu (90,7 % avant comme après).
>
> **3. Un seul split avec un seed fixe (plus ou moins 5 points).** Avec le code d'origine, changer `random_state` faisait varier le score de 83 % à 93 %. Un seul tirage de 193 tickets ne prouve rien. Remplacé par une validation croisée groupée, 5 plis × 5 répétitions : de 79,9 % à 80,8 %, donc stable.
>
> **4. Exactitude globale au lieu d'une mesure par catégorie.** La règle du README est par catégorie. Le chiffre global cachait une file facturation à 26 % d'erreur.
>
> **5. Étiquettes incohérentes dans les données.** La même demande est étiquetée différemment selon l'annotateur. Par exemple, « Le prix affiché n'est pas celui facturé » est étiqueté facturation, produit ou technique, et « Mon espace client affiche une erreur » technique ou compte. Cela concerne 20 demandes sur 78, soit 36 % des tickets (`analyse_donnees.py`). Le plafond est de 85 % même pour un modèle parfait, et ce sont exactement les confusions du ticket #88 (facturation ↔ livraison, compte → technique). Les doublons masquaient ce problème, car le modèle retenait le choix de l'annotateur. Ce n'est pas corrigeable dans le code : il faut un guide d'annotation décidé avec le service client.
>
> Les 2 points restants entre 80,3 % et les 78 % observés (47/60) sont dans le bruit : sur 60 tickets, 1 ticket représente 1,7 point.

## 2. Le vrai niveau

Donnez le vrai niveau de performance par catégorie. Quelles catégories respectent la règle métier du README, lesquelles non ?

> Validation croisée groupée par texte, sans doublons, sans seuil (`python evaluer.py`) : **80,3 % d'exactitude**, soit 19,7 % d'erreur, cohérent avec les 21,7 % constatés par le service client.
>
> | Catégorie | Précision | Rappel | Erreur dans la file | Règle < 10 % |
> |---|---|---|---|---|
> | facturation | 74,0 % | 78,1 % | 26,0 % | non |
> | technique | 77,0 % | 80,8 % | 23,0 % | non |
> | livraison | 83,0 % | 81,1 % | 17,0 % | non |
> | produit | 84,7 % | 78,8 % | 15,3 % | non |
> | compte | 86,1 % | 83,2 % | 13,9 % | non |
>
> **Aucune catégorie ne respecte la règle** : sans seuil, le modèle n'aurait jamais dû router seul. J'ai interprété la règle comme l'erreur dans la file reçue par une équipe (1 − précision), car c'est ce que l'équipe subit. Avec le seuil de 0,80, toutes les files passent sous 4 % (question 3).
>
> Limite : ce chiffre suppose que les nouveaux tickets sont des demandes déjà connues, formulées autrement. Sur un type de demande jamais vu, l'exactitude tombe à environ 52 %.

## 3. Le seuil

Quel seuil de confiance avez-vous retenu, et qu'est-ce que cela change concrètement pour le service client (combien de tickets restent en manuel, combien d'erreurs en moins) ?

> La confiance est la probabilité de la catégorie la plus probable (`predict_proba`). Sous le seuil, `predire.py` renvoie `a_trier`. Le tableau (`python seuils.py`) est calculé sur des prédictions hors échantillon, jamais sur les données d'entraînement :
>
> | Seuil | Routés auto | Erreur sur routés | Pire file (moyenne) | Pire file (pire tirage) | Erreurs évitées |
> |---|---|---|---|---|---|
> | aucun | 100 % | 19,7 % | facturation 26,0 % | | 0 % |
> | 0,60 | 78,1 % | 11,7 % | technique 14,7 % | 18,1 % | 53,5 % |
> | 0,70 | 67,4 % | 6,5 % | technique 9,4 % | 11,8 % | 77,8 % |
> | 0,75 | 62,2 % | 4,6 % | technique 6,8 % | 11,3 % | 85,5 % |
> | **0,80** | **54,4 %** | **2,1 %** | **technique 3,7 %** | **5,1 %** | **94,1 %** |
> | 0,85 | 37,9 % | 0,6 % | technique 2,1 % | 4,9 % | 98,9 % |
>
> **Seuil retenu : 0,80.** C'est le plus petit seuil (donc le plus de routage automatique) pour lequel chaque file reste sous 10 % d'erreur, y compris dans le pire des 5 tirages. 0,70 respecte la règle en moyenne (9,4 %) mais casse selon le tirage (11,8 %) : trop fragile pour des tickets inédits.
>
> Concrètement, sur une journée de 60 tickets comme celle du ticket #88 : environ 33 tickets sont routés automatiquement, 27 restent en manuel (`a_trier`), et on passe d'environ 12 tickets mal routés à moins de 1 (94 % des erreurs évitées). Le coût est de trier à la main 46 % des tickets, surtout les demandes ambiguës de la question 1, justement celles où le modèle se trompait. Si le service client accepte plus de risque, `--seuil` permet de changer ce réglage sans toucher au code.

## 4. L'IA

Donnez un exemple précis d'une proposition de votre assistant IA que vous avez refusée ou modifiée, et expliquez pourquoi. Si vous n'avez pas utilisé d'IA, dites-le.

> J'ai utilisé Claude. Après la correction des doublons, il m'a présenté les étiquettes incohérentes comme traitées : `evaluer.py` affichait « 3 textes identiques portent des étiquettes différentes ». J'ai remis ça en question, parce que 3 textes ne pouvaient pas expliquer les confusions décrites dans le ticket #88. En creusant, la détection ne voyait que les textes strictement identiques. Une fois les formules de politesse et les numéros de commande retirés, c'étaient en fait 20 demandes et 36 % des tickets, avec un plafond de 85 %. J'ai fait ajouter `analyse_donnees.py` pour le mesurer.
>
> Deuxième exemple : sa première analyse proposait un seuil de 0,70, car la règle était respectée en moyenne. J'ai retenu 0,80, car à 0,70 la file technique dépasse 10 % dans un des tirages : la marge est trop faible pour des tickets jamais vus. J'ai aussi refusé de recevoir tout `evaluer.py` corrigé d'un seul bloc, et demandé de traiter les problèmes un par un, avec un commit chacun.

## 5. Demain

Qu'est-ce qui, dans la vraie vie du service client, ferait baisser la performance de votre modèle sans que personne ne s'en aperçoive ? Comment le détecter ?

> **Ce qui ferait baisser la performance en silence :**
> - de nouveaux produits (un nouveau thé), de nouveaux problèmes (nouveau transporteur, nouvelle fonction du site) ou un pic saisonnier (livraisons à Noël) : le vocabulaire et la répartition des tickets changent ;
> - un changement de format des tickets : nouvel outil, signature automatique, HTML ;
> - des annotateurs qui n'appliquent pas tous la même règle.
>
> Personne ne s'en aperçoit, parce qu'une erreur sûre d'elle est routée sans alerte, et l'équipe la redirige à la main sans que ce soit compté.
>
> **Comment le détecter :**
> - suivre chaque semaine la part de tickets `a_trier` et la distribution des confiances. Sur des types de demandes jamais vus, le modèle n'est juste qu'à environ 52 %, mais il n'en route que 4 % : une hausse de `a_trier` est une alarme précoce ;
> - enregistrer chaque re-routage manuel comme une étiquette corrigée. On en tire l'erreur réelle par file chaque semaine, et c'est le seul signal qui attrape les erreurs sûres d'elles ;
> - surveiller la répartition des catégories prédites ;
> - chaque mois, étiqueter un échantillon de tickets récents avec un guide d'annotation commun et relancer `python evaluer.py --test nouveaux.csv`. Si une file dépasse 10 %, on remonte le seuil ou on ré-entraîne.