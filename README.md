# Word2vec
## Word Embedding : Implémentation de l'architecture CBOW
=====

# Modèle "word2vec" pour la construction de vecteurs de mots

Les représentations vectorielles de mots sont omniprésentes dans les systèmes de TAL actuels. Le modèle word2vec (Mikolov et al.
2013a et 2013b) notamment a eu un succès considérable. Dans ce projet on propose d'implémenter de
manière efficace le modèle word2vec CBOW, en utilisant la librairie pytorch.Si l'algorithme de
calcul des vecteurs est relativement simple (sachant que la partie optimisation est gérée via pytorch), il y a
un enjeu d'efficacité important. Il s'agira d'utiliser le plus possible une implémentation matricielle
pour accélérer les traitements et pouvoir apprendre sur des corpus de taille conséquente.

## TODO

- [ ] configuration et preproccessing
  - [x] class pour le preproccessing
  - [ ] implémentation de nouveaux hyperparamètres:
    - [x] pouvoir diviser le corpus à la taille souhaitée par l'utilisateur
    - [ ] implémenter un toy corpus par défault
    - [ ] avec ou sans lemmatisation
  - [ ] deplacer la création des batchs dans la class preproccess

- [x] ecrire le modèle
- [ ] affiner le model après test
- [ ] écrire une class dédiée au traitements des données entrainées:
  - [ ] stockage du model après entrainement
  - [ ] cluster plot
  - [ ] graphe de train
  - [ ] accuracy
- [ ] faire quelques logs
  - [ ] preproccess
  - [ ] CBOW

- [ ] constituer le dossier du projet

## Références

- [Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Thomas Mikolov et al 2013: Distributed Representations of Words and Phrases and their Compositionality. NIPS'13 Proceedings of the 26th International Conference on Neural Information Processing Systems](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
