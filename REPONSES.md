# Réponses

Répondez en quelques lignes par question, en français, avec vos mots.

## 1. L'écart

Quelles sont les causes de l'écart entre les 91 % annoncés et la réalité ? Pour chacune : où vous l'avez vue, et combien elle pèse.

> L'écart entre les 91 % annoncés et les résultats réels vient surtout des doublons : certains tickets étaient présents à la fois dans l'entraînement et le test. Après correction, le score passe à 80,3 %. Il y avait aussi une fuite de données avec le TF-IDF, un seul découpage des données et des catégories parfois incohérentes. La règle métier n'était pas non plus vérifiée catégorie par catégorie.


## 2. Le vrai niveau

Donnez le vrai niveau de performance par catégorie. Quelles catégories respectent la règle métier du README, lesquelles non ?

> Après une évaluation plus fiable, le modèle obtient 80,3 % d'exactitude. Sans seuil, aucune catégorie ne respecte la règle des 10 % d'erreurs : facturation atteint 26 %, technique 23 %, livraison 17 %, produit 15,3 % et compte 13,9 %. Le modèle ne pouvait donc pas router tous les tickets automatiquement.


## 3. Le seuil

Quel seuil de confiance avez-vous retenu, et qu'est-ce que cela change concrètement pour le service client (combien de tickets restent en manuel, combien d'erreurs en moins) ?

> J'ai retenu un seuil de confiance de 0,80. Avec ce seuil, 54,4 % des tickets sont routés automatiquement et l'erreur parmi ces tickets est de 2,1 %. Les 45,6 % restants sont laissés en manuel pour éviter les erreurs de routage. Ce choix permet de respecter la règle des 10 % dans toutes les files sur les évaluations réalisées.


## 4. L'IA

Donnez un exemple précis d'une proposition de votre assistant IA que vous avez refusée ou modifiée, et expliquez pourquoi. Si vous n'avez pas utilisé d'IA, dites-le.

> J'ai utilisé Claude pour m'aider à analyser les données et corriger le code. J'ai vérifié ses propositions, car la détection initiale ne signalait que trois textes incohérents, ce qui ne correspondait pas aux problèmes observés. J'ai approfondi l'analyse et choisi un seuil de 0,80 plutôt que 0,70, car il était plus fiable selon les résultats des différents tirages.

## 5. Demain

Qu'est-ce qui, dans la vraie vie du service client, ferait baisser la performance de votre modèle sans que personne ne s'en aperçoive ? Comment le détecter ?

> Les performances peuvent baisser si les produits, les demandes des clients ou le format des tickets changent. Pour le détecter, je surveillerais chaque semaine les tickets laissés en manuel et les erreurs corrigées par les agents. Je testerais aussi régulièrement le modèle sur de nouveaux tickets pour vérifier que chaque catégorie respecte toujours la règle des 10 %.