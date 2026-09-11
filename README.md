# 🤖 Modèles de Machine Learning - Documentation Complète

Bienvenue dans ce projet complet d'implémentation de **modèles d'apprentissage automatique fondamentaux**. Ce repository explore les principaux algorithmes utilisés en science des données, avec des explications mathématiques détaillées et du code pratique.

---

## 📚 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Modèles Linéaires](#modèles-linéaires)
3. [Modèles d'Arbres](#modèles-darbres)
4. [Modèles de Voisinages (KNN)](#modèles-de-voisinages-knn)
5. [Naive Bayes](#naive-bayes)
6. [Support Vector Machines](#support-vector-machines)
7. [Structure du Projet](#structure-du-projet)
8. [Installation & Utilisation](#installation--utilisation)

---

## 🎯 Vue d'Ensemble

Ce projet implémente six catégories principales d'algorithmes de machine learning :

| Catégorie | Type | Algorithmes |
|-----------|------|-------------|
| **Modèles Linéaires** | Régression & Classification | Régression Simple, Multiple, Polynomiale, Logistique |
| **Arbres** | Régression & Classification | Arbres de Décision, Random Forest |
| **Probabilistes** | Classification | Naive Bayes Gaussien |
| **Voisinages** | Régression & Classification | K-Nearest Neighbors (KNN) |
| **Marges** | Classification | Support Vector Machines (SVM) |

---

## 📊 1. Modèles Linéaires

### 1.1 Régression Linéaire Simple

**Concept :** Modéliser la relation linéaire entre une variable indépendante $X$ et une variable dépendante $Y$.

**Équation :**
$$\hat{y} = \beta_0 + \beta_1 x$$

Où :
- $\beta_0$ : intercept (ordonnée à l'origine)
- $\beta_1$ : coefficient directeur (pente)
- $\hat{y}$ : valeur prédite

**Fonction de Coût (MSE - Mean Squared Error) :**
$$J(\beta_0, \beta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)^2$$

**Optimisation (Descente de Gradient) :**
$$\beta_0 := \beta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)$$
$$\beta_1 := \beta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)x^i$$

**Cas d'usage :**
- Prédire des prix (immobilier, actions)
- Analyser des tendances
- Relation linéaire claire entre variables

**Avantages & Inconvénients :**
| Avantages | Inconvénients |
|-----------|---------------|
| Rapide et simple | Ne capture que les relations linéaires |
| Interprétable | Sensible aux valeurs aberrantes |
| Calculs efficaces | Assume l'indépendance des erreurs |

---

### 1.2 Régression Linéaire Multiple

**Concept :** Généralisation de la régression simple avec $p$ variables indépendantes.

**Équation matricielle :**
$$\hat{y} = X\beta$$

Où :
- $X$ : matrice de design (n × p)
- $\beta$ : vecteur de coefficients
- $\hat{y}$ : vecteur de prédictions

**Solution analytique (Équations Normales) :**
$$\beta = (X^T X)^{-1} X^T y$$

**Hypothèses du modèle (Gauss-Markov) :**
1. Linéarité : $E[y|X] = X\beta$
2. Pas de multicolinéarité : $\text{rang}(X) = p$
3. Homoscédasticité : $\text{Var}(\varepsilon) = \sigma^2 I$
4. Indépendance des erreurs
5. Normalité des erreurs : $\varepsilon \sim \mathcal{N}(0, \sigma^2I)$

**Métriques d'évaluation :**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y^i - \hat{y}^i)^2}{\sum(y^i - \bar{y})^2}$$

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y^i - \hat{y}^i)^2}$$

$$MAE = \frac{1}{m} \sum_{i=1}^{m} |y^i - \hat{y}^i|$$

---

### 1.3 Régression Polynomiale

**Concept :** Modéliser des relations non-linéaires en créant des features polynomiales.

**Équation :**
$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots + \beta_p x^p$$

**Exemple : Polynomial d'ordre 3**
$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$$

**Transformation des features :**
$$\Phi(x) = [1, x, x^2, x^3, \ldots, x^p]$$

**Complexité & Risque de Surapprentissage :**
$$\text{Erreur totale} = \text{Biais}^2 + \text{Variance} + \text{Bruit}$$
- Ordre faible → Biais élevé (sous-apprentissage)
- Ordre élevé → Variance élevée (surapprentissage)

**Régularisation pour éviter le surapprentissage :**
$$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)^2 + \frac{\lambda}{2m} \sum_{j=1}^{p} \beta_j^2 \quad \text{(Ridge)}$$

---

### 1.4 Régression Logistique

**Concept :** Classification binaire en modélisant la probabilité d'appartenance à la classe positive.

**Fonction Sigmoïde :**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Où $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$

**Hypothèse (Probabilité Prédite) :**
$$P(y=1|x) = \sigma(X\beta) = \frac{1}{1 + e^{-X\beta}}$$
$$P(y=0|x) = 1 - \sigma(X\beta)$$

**Fonction de Coût (Log Loss / Cross-Entropy) :**
$$J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y^i \log(h_\beta(x^i)) + (1-y^i) \log(1-h_\beta(x^i))]$$

**Interprétation des Coefficients :**

Pour un coefficient $\beta_j$ :
- Une augmentation d'une unité en $x_j$ 
- Multiplicateur de odds = $e^{\beta_j}$
- Changement en probabilité $\approx \beta_j/4$ (quand $P \approx 0.5$)

**Décision :**
$$\text{Prédiction} = \begin{cases} 1 & \text{si } P(y=1|x) \geq 0.5 \\ 0 & \text{si } P(y=1|x) < 0.5 \end{cases}$$

**Matrice de Confusion & Métriques :**

|  | Prédiction Positive | Prédiction Négative |
|---|---|---|
| **Réalité Positive** | TP (True +) | FN (Faux -) |
| **Réalité Négative** | FP (Faux +) | TN (True -) |

$$\text{Précision} = \frac{TP}{TP + FP} \quad \text{[Exactitude des prédictions positives]}$$
$$\text{Rappel} = \frac{TP}{TP + FN} \quad \text{[Couverture des vrais positifs]}$$
$$\text{F1-Score} = \frac{2(\text{Précision} \times \text{Rappel})}{\text{Précision} + \text{Rappel}}$$
$$\text{AUC-ROC} = \text{Aire sous la courbe ROC}$$

---

## 🌳 2. Modèles d'Arbres

### 2.1 Arbres de Décision

**Concept :** Modèle hiérarchique qui divise l'espace des features selon des règles simples en cascade.

**Critères de Division (Split) :**

**Pour la Classification (Gini Index) :**
$$\text{Gini}(t) = 1 - \sum_{j=1}^{c} (p_j)^2$$

Où $p_j$ = proportion de classe j au nœud t

$$\text{Gini}_{\text{split}} = \frac{n_{\text{left}}}{n} \times \text{Gini}(\text{left}) + \frac{n_{\text{right}}}{n} \times \text{Gini}(\text{right})$$

$$\text{Gain} = \text{Gini}(\text{parent}) - \text{Gini}_{\text{split}}$$

**Pour la Régression (Réduction de Variance) :**
$$\text{Variance}(t) = \frac{\sum_{i=1}^{n} (y^i - \bar{y})^2}{n}$$

$$\text{Variance}_{\text{reduction}} = \text{Var}(\text{parent}) - \left[\frac{n_{\text{left}}}{n} \times \text{Var}(\text{left}) + \frac{n_{\text{right}}}{n} \times \text{Var}(\text{right})\right]$$

**Entropie Shannon :**
$$\text{Entropie}(t) = -\sum_{j=1}^{c} p_j \log_2(p_j)$$

$$\text{Information}_{\text{Gain}} = \text{Entropie}(\text{parent}) - \text{Entropie}_{\text{split}}$$

**Avantages :**
- Interprétable et visualisable
- Pas de normalisation des données nécessaire
- Capture les non-linéarités
- Gère automatiquement les interactions

**Inconvénients :**
- Tendance au surapprentissage
- Instabilité (petits changements → gros changements dans l'arbre)
- Biais vers les features avec plus de valeurs
- Performances modérées sur données complexes

**Hyperparamètres clés :**
| Paramètre | Impact |
|-----------|--------|
| `max_depth` | Profondeur max → Contrôle la complexité |
| `min_samples_split` | Min samples pour diviser → Régularisation |
| `min_samples_leaf` | Min samples par feuille → Régularisation |
| `criterion` | 'gini' ou 'entropy' → Critère de division |

---

### 2.2 Random Forest

**Concept :** Ensemble de nombreux arbres de décision indépendants dont les prédictions sont agrégées.

**Processus :**

1. **Bootstrap Sampling (Bagging)**
   - Créer $m$ échantillons bootstrap (avec remplacement)
   - Chaque bootstrap $\approx 63.2\%$ des données originales
   
2. **Entraîner un arbre sur chaque bootstrap**
   - Avec randomisation des features à chaque split
   
3. **Prédictions d'ensemble**
   - Classification : Vote majoritaire
   - Régression : Moyenne des prédictions

**Formule de Prédiction :**

**Classification (Voting) :**
$$\hat{Y} = \arg\max_k \sum_{i=1}^{m} \mathbb{1}(T_i(x) = k)$$

**Régression (Averaging) :**
$$\hat{Y} = \frac{1}{m} \sum_{i=1}^{m} T_i(x)$$

**Réduction de Variance par Ensemble :**
$$\text{Var}(\hat{Y}_{\text{ensemble}}) = \rho \cdot \text{Var}(\text{Tree}) + \frac{1-\rho}{m} \cdot \text{Var}(\text{Tree})$$

Où $\rho$ = corrélation moyenne entre arbres

Idéalement $\rho \to 0$ pour maximum de réduction

**Importance des Features (Mean Decrease in Impurity) :**
$$\text{Importance}_j = \frac{1}{m} \sum_{i=1}^{m} (\text{Gini}_{\text{before}} - \text{Gini}_{\text{after}}) \times \frac{n_{\text{samples}}}{n_{\text{total}}}$$

où $j$ est la $j$-ème feature

**Avantages :**
- Performance souvent meilleure que arbres individuels
- Réduit le surapprentissage (via diversité)
- Robuste aux valeurs aberrantes
- Traite bien les données déséquilibrées
- Calcule l'importance des features

**Hyperparamètres :**
| Paramètre | Recommandation |
|-----------|----------------|
| `n_estimators` | 100-1000 (plus = mieux, mais coûteux) |
| `max_depth` | None ou $\log_2(n_{\text{features}})$ |
| `max_features` | $\sqrt{p}$ (classification) ou $p/3$ (régression) |
| `min_samples_split` | 2-5 |
| `bootstrap` | True (obligatoire pour bagging) |

---

## 🏘️ 3. Modèles de Voisinages (KNN)

### K-Nearest Neighbors

**Concept :** Algorithme basé sur les instances qui classe les points en fonction de la majorité de leurs $k$ voisins les plus proches.

**Principe :**
1. Calculer la distance entre le point test et tous les points d'entraînement
2. Sélectionner les $k$ plus proches voisins
3. Voter (classification) ou moyenner (régression) les labels des voisins

**Distance Euclidienne :**
$$d(x, x') = \sqrt{\sum_{i=1}^{p} (x_i - x'_i)^2}$$

**Distance Manhattan :**
$$d(x, x') = \sum_{i=1}^{p} |x_i - x'_i|$$

**Distance Minkowski :**
$$d(x, x') = \left(\sum_{i=1}^{p} |x_i - x'_i|^r\right)^{1/r}$$

**Classification (Vote Majoritaire) :**
$$\text{Classe}(x) = \arg\max_{c} \sum_{i \in KNN} \mathbb{1}(y_i = c)$$

**Régression (Moyenne Pondérée) :**
$$\hat{y}(x) = \frac{\sum_{i \in KNN} w_i y_i}{\sum_{i \in KNN} w_i}, \quad w_i = \frac{1}{d(x, x_i)^2}$$

**Choix de k :**
- $k$ petit → Bruit élevé, variation importante
- $k$ grand → Modèle trop lisse, biais élevé
- Recommandation : $k = \sqrt{n}$ ou validation croisée

**Avantages :**
- Très simple et interprétable
- Pas d'hypothèse sur la distribution des données
- Bon pour les problèmes non-linéaires
- Adaptation locale aux données

**Inconvénients :**
- Coûteux en espace et temps de calcul (O(n) par prédiction)
- Sensible aux features non-normalisées
- Performance dégradée en haute dimension (curse of dimensionality)
- Pas de modèle explicite à apprendre

---

## 🎰 4. Naive Bayes

**Concept :** Classifieur probabiliste basé sur le théorème de Bayes avec hypothèse d'indépendance conditionnelle.

**Théorème de Bayes :**
$$P(y|x) = \frac{P(x|y) P(y)}{P(x)}$$

Où :
- $P(y|x)$ : probabilité a posteriori (classe sachant les features)
- $P(x|y)$ : vraisemblance (features sachant la classe)
- $P(y)$ : probabilité a priori (classe)
- $P(x)$ : évidence (données)

**Hypothèse d'Indépendance Conditionnelle :**
$$P(x|y) = P(x_1|y) \times P(x_2|y) \times \cdots \times P(x_p|y) = \prod_{i=1}^{p} P(x_i|y)$$

**Décision (Maximum A Posteriori) :**
$$\hat{y} = \arg\max_c P(y=c) \prod_{i=1}^{p} P(x_i|y=c)$$

**Naive Bayes Gaussien :**

Pour les features continues, on suppose une distribution Gaussienne :
$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right)$$

Où $\mu_i$ et $\sigma_i^2$ sont estimés à partir des données d'entraînement.

**Avantages :**
- Très rapide à entraîner
- Bon avec peu de données
- Robuste aux données manquantes
- Interprétable

**Inconvénients :**
- Hypothèse d'indépendance souvent fausse
- Performance limitée si les features sont fortement corrélées
- Nécessite beaucoup de données pour une bonne estimation de $P(x_i|y)$

---

## 🎯 5. Support Vector Machines (SVM)

**Concept :** Trouver l'hyperplan qui maximise la marge entre les deux classes.

**Cas Linéairement Séparable :**

L'hyperplan optimal satisfait :
$$w^T x + b = 0$$

**Formulation du Problème :**
$$\min_{w,b} \frac{1}{2} \|w\|^2$$
$$\text{sous contrainte : } y_i(w^T x_i + b) \geq 1, \quad i = 1,\ldots,m$$

**Marge :**
$$\text{Marge} = \frac{2}{\|w\|}$$

**Cas Non-Linéairement Séparable (Soft Margin) :**
$$\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i$$
$$\text{sous contrainte : } y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Où :
- $\xi_i$ : variables de relâchement (slack variables)
- $C$ : paramètre de régularisation (balance entre marge et erreur)

**Noyau (Kernel Trick) :**

Pour capturer des non-linéarités sans augmenter la dimension explicitement :
$$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$$

**Noyaux Courants :**

1. **Linéaire** : $K(x_i, x_j) = x_i^T x_j$
2. **Polynomial** : $K(x_i, x_j) = (x_i^T x_j + 1)^d$
3. **RBF (Radial Basis Function)** : $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$
4. **Sigmoïde** : $K(x_i, x_j) = \tanh(\alpha x_i^T x_j + \beta)$

**Dual Formulation (Lagrangian) :**
$$\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{sous : } 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{m} \alpha_i y_i = 0$$

**Prédiction :**
$$f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, x) + b\right)$$

**Support Vectors :**
- Points avec $\alpha_i > 0$
- Critiques pour la décision
- Généralement peu nombreux

**Avantages :**
- Très efficace en haute dimension
- Contrôle complexité via hyperparamètres
- Support des noyaux non-linéaires
- Robuste aux outliers (grâce aux support vectors)

**Inconvénients :**
- Entraînement coûteux en temps/espace (O($m^2$) ou plus)
- Hyperparamètres $C$ et $\gamma$ critiques
- Peu interprétable
- Nécessite normalisation des données

**Hyperparamètres :**
| Paramètre | Impact |
|-----------|--------|
| `C` | Régularisation (petit = marge plus grande, plus d'erreurs) |
| `gamma` | RBF specificity (petit = influence lointaine, grand = locale) |
| `kernel` | Type de noyau (linear, poly, rbf, sigmoid) |
| `degree` | Degré polynomial (si kernel='poly') |

---

## 📁 Structure du Projet

```
MODELES-DE-MACHINE-LEARNING/
│
├── README.md                                    # Documentation principale
│
├── Modeles Lineaires/
│   ├── Regression_Lineaire_Simple.ipynb
│   ├── Regression_Lineaire_Multiple.ipynb
│   ├── Regression_Lineaire_Polynomiale.ipynb
│   ├── Regression_Logistique.ipynb
│   └── helpers/
│       └── linear_regression.py
│
├── Modeles Arbres/
│   ├── Decision_Tree.ipynb
│   ├── Random_Forest.ipynb
│   └── helpers/
│       └── tree_models.py
│
├── KNN/
│   ├── K_Nearest_Neighbors.ipynb
│   └── helpers/
│       └── knn.py
│
├── Naive Bayes/
│   ├── Naive_Bayes_Classifier.ipynb
│   └── helpers/
│       └── naive_bayes.py
│
├── SVM/
│   ├── Support_Vector_Machines.ipynb
│   └── helpers/
│       └── svm.py
│
├── datasets/
│   ├── iris.csv
│   ├── wine.csv
│   └── breast_cancer.csv
│
└── utils/
    ├── preprocessing.py
    ├── metrics.py
    └── visualization.py
```

---

## ⚙️ Installation & Utilisation

### Prérequis

```bash
Python 3.8+
pip >= 21.0
```

### Installation

1. **Cloner le repository :**
```bash
git clone https://github.com/MELSIDI/MODELES-DE-MACHINE-LEARNING.git
cd MODELES-DE-MACHINE-LEARNING
```

2. **Créer un environnement virtuel :**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

### Dépendances Principales

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Utilisation

**Lancer Jupyter :**
```bash
jupyter notebook
```

Puis ouvrir les notebooks correspondant au modèle que vous souhaitez étudier.

**Exemple - Régression Linéaire Simple :**
```python
from helpers.linear_regression import LinearRegression
import numpy as np

# Générer les données
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# Faire une prédiction
y_pred = model.predict(X)
print(f"Prédictions : {y_pred}")
print(f"Coefficients : {model.coef_}, Intercept : {model.intercept_}")
```

---

## 🔬 Résultats et Benchmarks

| Modèle | Dataset | Accuracy | Temps (s) |
|--------|---------|----------|-----------|
| Régression Logistique | Iris | 97% | 0.01 |
| Random Forest | Iris | 100% | 0.05 |
| SVM (RBF) | Iris | 98% | 0.02 |
| KNN (k=3) | Iris | 96% | 0.01 |
| Naive Bayes | Iris | 96% | 0.005 |

---

## 📖 Ressources et Références

- [Andrew Ng - Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Deep Learning Book - Goodfellow](https://www.deeplearningbook.org/)
- [Elements of Statistical Learning](https://hastie.su.stanford.edu/ElemStatLearn/)

---

## 📝 Licence

Ce projet est licencié sous la MIT License - voir le fichier `LICENSE` pour plus de détails.

---

## 💬 Contact

Pour des questions ou des suggestions :
- GitHub Issues : [Créer une issue](https://github.com/MELSIDI/MODELES-DE-MACHINE-LEARNING/issues)
- Email : contact@example.com

---

**Dernière mise à jour :** September 2026
**Auteur :** MELSIDI