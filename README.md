<div align="center">

# 🤖 Modèles de Machine Learning — Documentation Complète

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)


Implémentation et documentation mathématique des **algorithmes fondamentaux du Machine Learning**, avec explications théoriques détaillées et code pratique associé.

</div>

---

## 📚 Table des matières
 
1. [Vue d'ensemble](#-1-vue-densemble)
2. [Modèles linéaires](#-2-modèles-linéaires)
   - 2.1 [Régression linéaire simple](#21-régression-linéaire-simple)
   - 2.2 [Régression linéaire multiple](#22-régression-linéaire-multiple)
   - 2.3 [Régression polynomiale](#23-régression-polynomiale)
   - 2.4 [Régression logistique](#24-régression-logistique)
3. [Modèles d'arbres](#-3-modèles-darbres)
   - 3.1 [Arbres de décision](#31-arbres-de-décision)
   - 3.2 [Random Forest](#32-random-forest)
4. [Modèles de voisinage (KNN)](#-4-modèles-de-voisinage-knn)
5. [Naive Bayes](#-5-naive-bayes)
6. [Support Vector Machines (SVM)](#-6-support-vector-machines-svm)
7. [Résultats et benchmarks](#-7-résultats-et-benchmarks)
8. [Installation & utilisation](#-8-installation--utilisation)
---
 
## 🎯 1. Vue d'ensemble
 
Ce projet implémente six familles principales d'algorithmes de Machine Learning :
 
| Catégorie | Type | Algorithmes |
|---|---|---|
| **Modèles linéaires** | Régression & Classification | Simple, Multiple, Polynomiale, Logistique |
| **Arbres** | Régression & Classification | Arbre de décision, Random Forest |
| **Probabilistes** | Classification | Naive Bayes Gaussien |
| **Voisinage** | Régression & Classification | K-Nearest Neighbors (KNN) |
| **Marges** | Classification | Support Vector Machines (SVM) |
 
---
 
## 📊 2. Modèles linéaires
 
### 2.1 Régression linéaire simple
 
**Concept.** Modéliser la relation linéaire entre une variable indépendante $X$ et une variable dépendante $Y$.
 
**Équation :**
 
$$
\hat{y} = \beta_0 + \beta_1 x
$$
 
- $\beta_0$ : intercept (ordonnée à l'origine)
- $\beta_1$ : coefficient directeur (pente)
- $\hat{y}$ : valeur prédite
**Fonction de coût (MSE) :**
 
$$
J(\beta_0, \beta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)^2
$$
 
**Optimisation (descente de gradient) :**
 
$$
\beta_0 := \beta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)
$$
 
$$
\beta_1 := \beta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)x^i
$$
 
**Cas d'usage :** prédiction de prix (immobilier, actions), analyse de tendances, relation linéaire claire entre variables.
 
| Avantages | Inconvénients |
|---|---|
| Rapide et simple | Ne capture que les relations linéaires |
| Interprétable | Sensible aux valeurs aberrantes |
| Calculs efficaces | Assume l'indépendance des erreurs |
 
---
 
### 2.2 Régression linéaire multiple
 
**Concept.** Généralisation de la régression simple à $p$ variables indépendantes.
 
**Équation matricielle :**
 
$$
\hat{y} = X\beta
$$
 
- $X$ : matrice de design ($n \times p$)
- $\beta$ : vecteur de coefficients
- $\hat{y}$ : vecteur de prédictions
**Solution analytique (équations normales) :**
 
$$
\beta = (X^T X)^{-1} X^T y
$$
 
**Hypothèses du modèle (Gauss-Markov) :**
1. Linéarité : $E[y|X] = X\beta$
2. Pas de multicolinéarité : $\text{rang}(X) = p$
3. Homoscédasticité : $\text{Var}(\varepsilon) = \sigma^2 I$
4. Indépendance des erreurs
5. Normalité des erreurs : $\varepsilon \sim \mathcal{N}(0, \sigma^2I)$
**Métriques d'évaluation :**
 
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y^i - \hat{y}^i)^2}{\sum(y^i - \bar{y})^2}
$$
 
$$
RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y^i - \hat{y}^i)^2} \qquad MAE = \frac{1}{m} \sum_{i=1}^{m} |y^i - \hat{y}^i|
$$
 
---
 
### 2.3 Régression polynomiale
 
**Concept.** Modéliser des relations non linéaires en créant des features polynomiales.
 
**Équation (ordre $p$) :**
 
$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots + \beta_p x^p
$$
 
**Transformation des features :**
 
$$
\Phi(x) = [1, x, x^2, x^3, \ldots, x^p]
$$
 
**Compromis biais–variance :**
 
$$
\text{Erreur totale} = \text{Biais}^2 + \text{Variance} + \text{Bruit}
$$
 
- Ordre faible → biais élevé (sous-apprentissage)
- Ordre élevé → variance élevée (surapprentissage)
**Régularisation (Ridge) :**
 
$$
J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^i) - y^i)^2 + \frac{\lambda}{2m} \sum_{j=1}^{p} \beta_j^2
$$
 
---
 
### 2.4 Régression logistique
 
**Concept.** Classification binaire par modélisation de la probabilité d'appartenance à la classe positive.
 
**Fonction sigmoïde :**
 
$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$
 
**Probabilités prédites :**
 
$$
P(y=1|x) = \sigma(X\beta) = \frac{1}{1 + e^{-X\beta}} \qquad P(y=0|x) = 1 - \sigma(X\beta)
$$
 
**Fonction de coût (log loss) :**
 
$$
J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y^i \log(h_\beta(x^i)) + (1-y^i) \log(1-h_\beta(x^i))]
$$
 
**Interprétation des coefficients.** Pour $\beta_j$ : une augmentation d'une unité de $x_j$ multiplie les odds par $e^{\beta_j}$ (et change la probabilité d'environ $\beta_j/4$ quand $P \approx 0.5$).
 
**Règle de décision :**
 
$$
\text{Prédiction} = 1 \text{ si } P(y=1|x) \geq 0.5, \quad \text{sinon } 0
$$
 
**Matrice de confusion :**
 
|  | Prédit + | Prédit − |
|---|---|---|
| **Réel +** | TP | FN |
| **Réel −** | FP | TN |
 
**Métriques :**
 
$$
\text{Précision} = \frac{TP}{TP + FP} \qquad \text{Rappel} = \frac{TP}{TP + FN} \qquad F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$
 
---
 
## 🌳 3. Modèles d'arbres
 
### 3.1 Arbres de décision
 
**Concept.** Modèle hiérarchique qui divise l'espace des features selon des règles simples en cascade.
 
**Critère de split — Classification (indice de Gini) :**
 
$$
\text{Gini}(t) = 1 - \sum_{j=1}^{c} p_j^2
$$
 
$$
\text{Gini}_{\text{split}} = \frac{n_{\text{left}}}{n} \text{Gini}(\text{left}) + \frac{n_{\text{right}}}{n} \text{Gini}(\text{right}) \qquad \text{Gain} = \text{Gini}(\text{parent}) - \text{Gini}_{\text{split}}
$$
 
**Critère de split — Régression (réduction de variance) :**
 
$$
\text{Var}(t) = \frac{1}{n}\sum_{i=1}^{n} (y^i - \bar{y})^2
$$
 
$$
\Delta\text{Var} = \text{Var}(\text{parent}) - \left[\frac{n_{\text{left}}}{n} \text{Var}(\text{left}) + \frac{n_{\text{right}}}{n} \text{Var}(\text{right})\right]
$$
 
**Entropie de Shannon :**
 
$$
\text{Entropie}(t) = -\sum_{j=1}^{c} p_j \log_2(p_j) \qquad \text{Gain d'information} = \text{Entropie}(\text{parent}) - \text{Entropie}_{\text{split}}
$$
 
| Avantages | Inconvénients |
|---|---|
| Interprétable et visualisable | Tendance au surapprentissage |
| Pas de normalisation nécessaire | Instable (sensible aux petites variations) |
| Capture les non-linéarités | Biais vers les features à forte cardinalité |
| Gère les interactions automatiquement | Performances modérées seul |
 
**Hyperparamètres clés :**
 
| Paramètre | Rôle |
|---|---|
| `max_depth` | Profondeur max → contrôle la complexité |
| `min_samples_split` | Min. d'échantillons pour diviser → régularisation |
| `min_samples_leaf` | Min. d'échantillons par feuille → régularisation |
| `criterion` | `gini` ou `entropy` |
 
---
 
### 3.2 Random Forest
 
**Concept.** Ensemble de nombreux arbres de décision indépendants dont les prédictions sont agrégées.
 
**Processus :**
1. **Bootstrap sampling (bagging)** — créer $m$ échantillons bootstrap (avec remise), chacun $\approx 63{,}2\%$ des données originales
2. **Entraîner un arbre par bootstrap**, avec randomisation des features à chaque split
3. **Agréger les prédictions** — vote majoritaire (classification) ou moyenne (régression)
**Prédiction :**
 
$$
\text{Classification : } \hat{Y} = \arg\max_k \sum_{i=1}^{m} \mathbb{1}(T_i(x) = k) \qquad \text{Régression : } \hat{Y} = \frac{1}{m} \sum_{i=1}^{m} T_i(x)
$$
 
**Réduction de variance par ensemble :**
 
$$
\text{Var}(\hat{Y}_{\text{ensemble}}) = \rho \cdot \text{Var}(\text{Tree}) + \frac{1-\rho}{m} \cdot \text{Var}(\text{Tree})
$$
 
où $\rho$ est la corrélation moyenne entre arbres (idéalement $\rho \to 0$).
 
**Importance des features (Mean Decrease in Impurity) :**
 
$$
\text{Importance}_j = \frac{1}{m} \sum_{i=1}^{m} (\text{Gini}_{\text{avant}} - \text{Gini}_{\text{après}}) \times \frac{n_{\text{samples}}}{n_{\text{total}}}
$$
 
| Avantages | Inconvénients |
|---|---|
| Meilleure performance qu'un arbre seul | Moins interprétable |
| Réduit le surapprentissage | Plus coûteux en calcul |
| Robuste aux outliers | Modèle plus lourd à stocker |
| Gère bien les données déséquilibrées | — |
 
**Hyperparamètres recommandés :**
 
| Paramètre | Recommandation |
|---|---|
| `n_estimators` | 100–1000 |
| `max_depth` | `None` ou $\log_2(n_{\text{features}})$ |
| `max_features` | $\sqrt{p}$ (classification) ou $p/3$ (régression) |
| `min_samples_split` | 2–5 |
| `bootstrap` | `True` |
 
---
 
## 🏘️ 4. Modèles de voisinage (KNN)
 
**Concept.** Algorithme à base d'instances qui classe un point selon la majorité de ses $k$ plus proches voisins.
 
**Principe :**
1. Calculer la distance entre le point test et tous les points d'entraînement
2. Sélectionner les $k$ plus proches voisins
3. Voter (classification) ou moyenner (régression) leurs labels
**Distances :**
 
$$
\text{Euclidienne : } d(x, x') = \sqrt{\sum_{i=1}^{p} (x_i - x'_i)^2} \qquad \text{Manhattan : } d(x, x') = \sum_{i=1}^{p} |x_i - x'_i|
$$
 
$$
\text{Minkowski : } d(x, x') = \left(\sum_{i=1}^{p} |x_i - x'_i|^r\right)^{1/r}
$$
 
**Classification (vote majoritaire) :**
 
$$
\text{Classe}(x) = \arg\max_{c} \sum_{i \in KNN} \mathbb{1}(y_i = c)
$$
 
**Régression (moyenne pondérée) :**
 
$$
\hat{y}(x) = \frac{\sum_{i \in KNN} w_i y_i}{\sum_{i \in KNN} w_i}, \quad w_i = \frac{1}{d(x, x_i)^2}
$$
 
**Choix de $k$.** Petit $k$ → forte variance/bruit ; grand $k$ → modèle trop lissé, biais élevé. Recommandation : $k = \sqrt{n}$ ou validation croisée.
 
| Avantages | Inconvénients |
|---|---|
| Simple et interprétable | Coûteux en temps/espace ($O(n)$ par prédiction) |
| Aucune hypothèse sur la distribution | Sensible aux features non normalisées |
| Bon sur problèmes non linéaires | Dégradé en haute dimension |
| Adaptation locale aux données | Pas de modèle explicite appris |
 
---
 
## 🎰 5. Naive Bayes
 
**Concept.** Classifieur probabiliste basé sur le théorème de Bayes avec hypothèse d'indépendance conditionnelle des features.
 
**Théorème de Bayes :**
 
$$
P(y|x) = \frac{P(x|y)\, P(y)}{P(x)}
$$
 
- $P(y|x)$ : probabilité a posteriori
- $P(x|y)$ : vraisemblance
- $P(y)$ : probabilité a priori
- $P(x)$ : évidence
**Hypothèse d'indépendance conditionnelle :**
 
$$
P(x|y) = \prod_{i=1}^{p} P(x_i|y)
$$
 
**Décision (Maximum A Posteriori) :**
 
$$
\hat{y} = \arg\max_c P(y=c) \prod_{i=1}^{p} P(x_i|y=c)
$$
 
**Naive Bayes Gaussien** (features continues) :
 
$$
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right)
$$
 
où $\mu_i$ et $\sigma_i^2$ sont estimés à partir des données d'entraînement.
 
| Avantages | Inconvénients |
|---|---|
| Très rapide à entraîner | Hypothèse d'indépendance souvent fausse |
| Efficace avec peu de données | Performance limitée si features corrélées |
| Robuste aux données manquantes | Estimation de $P(x_i|y)$ gourmande en données |
| Interprétable | — |
 
---
 
## 🎯 6. Support Vector Machines (SVM)
 
**Concept.** Trouver l'hyperplan qui maximise la marge entre les classes.
 
**Cas linéairement séparable :**
 
$$
w^T x + b = 0
$$
 
$$
\min_{w,b} \frac{1}{2} \|w\|^2 \quad \text{s.c. } y_i(w^T x_i + b) \geq 1,\ i = 1,\ldots,m
$$
 
$$
\text{Marge} = \frac{2}{\|w\|}
$$
 
**Cas non linéairement séparable (marge souple) :**
 
$$
\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i \quad \text{s.c. } y_i(w^T x_i + b) \geq 1 - \xi_i,\ \xi_i \geq 0
$$
 
où $\xi_i$ sont les variables de relâchement et $C$ le paramètre de régularisation.
 
**Astuce du noyau (kernel trick) :**
 
$$
K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle
$$
 
| Noyau | Formule |
|---|---|
| Linéaire | $K(x_i, x_j) = x_i^T x_j$ |
| Polynomial | $K(x_i, x_j) = (x_i^T x_j + 1)^d$ |
| RBF | $K(x_i, x_j) = \exp(-\gamma \lVert x_i - x_j \rVert^2)$ |
| Sigmoïde | $K(x_i, x_j) = \tanh(\alpha x_i^T x_j + \beta)$ |
 
**Formulation duale (lagrangienne) :**
 
$$
\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.c. } 0 \leq \alpha_i \leq C,\ \sum_{i=1}^{m} \alpha_i y_i = 0
$$
 
**Prédiction :**
 
$$
f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(x_i, x) + b\right)
$$
 
Les *support vectors* sont les points avec $\alpha_i > 0$ — généralement peu nombreux, et critiques pour la décision.
 
| Avantages | Inconvénients |
|---|---|
| Très efficace en haute dimension | Entraînement coûteux ($O(m^2)$ ou plus) |
| Contrôle fin de la complexité | Hyperparamètres $C$, $\gamma$ sensibles |
| Support de noyaux non linéaires | Peu interprétable |
| Robuste aux outliers | Nécessite la normalisation des données |
 
**Hyperparamètres :**
 
| Paramètre | Impact |
|---|---|
| `C` | Régularisation (petit = marge plus grande, plus d'erreurs tolérées) |
| `gamma` | Spécificité du noyau RBF (petit = influence lointaine, grand = locale) |
| `kernel` | `linear`, `poly`, `rbf`, `sigmoid` |
| `degree` | Degré polynomial (si `kernel='poly'`) |
 
---
 
## 🔬 7. Résultats et benchmarks
 
Résultats obtenus sur le dataset **Iris** (référence de comparaison rapide entre modèles) :
 
| Modèle | Dataset | Accuracy | Temps (s) |
|---|---|---|---|
| Régression logistique | Iris | 97% | 0.01 |
| Random Forest | Iris | 100% | 0.05 |
| SVM (RBF) | Iris | 98% | 0.02 |
| KNN (k=3) | Iris | 96% | 0.01 |
| Naive Bayes | Iris | 96% | 0.005 |
 
---
 
## ⚙️ 8. Installation & utilisation
 
### Prérequis
 
```bash
Python 3.8+
pip >= 21.0
```
 
### Installation
 
```bash
# 1. Cloner le repository
git clone https://github.com/MELSIDI/MODELES-DE-MACHINE-LEARNING.git
cd MODELES-DE-MACHINE-LEARNING
 
# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
 
# 3. Installer les dépendances
pip install -r requirements.txt
```
 
### Dépendances principales
 
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```
 
### Utilisation
 
```bash
jupyter notebook
```
 
Puis ouvrir le notebook correspondant au modèle à étudier.
 
**Exemple — Régression linéaire simple :**
 
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
 
