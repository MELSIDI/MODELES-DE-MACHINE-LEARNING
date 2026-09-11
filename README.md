# 🤖 Modèles de Machine Learning - Documentation Complète

Bienvenue dans ce projet complet d'implémentation de **modèles d'apprentissage automatique fondamentaux**. Ce repository explore les principaux algorithmes utilisés en science des données, avec[...]

---

## 📚 Table des Matires

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
-  Prédire des prix (immobilier, actions)
-  Analyser des tendances
-  Relation linéaire claire entre variables

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
-  Interprétable et visualisable
-  Pas de normalisation des données nécessaire
-  Capture les non-linéarités
-  Gère automatiquement les interactions

**Inconvénients :**
-  Tendance au surapprentissage
-  Instabilité (petits changements → gros changements dans l'arbre)
-  Biais vers les features avec plus de valeurs
-  Performances modérées sur données complexes

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
-  Performance souvent meilleure que arbres individuels
-  Réduit le surapprentissage (via diversité)
-  Robuste aux valeurs aberrantes
-  Traite bien les données déséquilibrées
-  Calcule l'importance des features

**Hyperparamètres :**
| Paramètre | Recommandation |
|-----------|----------------|
| `n_estimators` | 100-1000 (plus = mieux, mais coûteux) |
| `max_depth` | None ou $\log_2(n_{\text{features}})$ |
| `max_features` | $\sqrt{p}$ (classification) ou $p/3$ (régression) |
| `min_samples_split` | 2-5 |
| `bootstrap` | True (obligatoire pour bagging) |

---

##  3. Modèles de Voisinages (KNN)

### K-Nearest Neighbors

... (le reste du fichier inchangé) ...

## 📁 Structure du Projet

```
MODELES-DE-MACHINE-LEARNING/
│
├── README.md                           # Documentation principale (ce fichier)
│
├── Modeles Lineaires/                  # Linear models (notebooks + helpers)
│   ├── Regression_Linieaire_Simple.ipynb
│   ├── Regression_Lineaire_Polynomiale.ipynb
{