# 🤖 Modèles de Machine Learning - Documentation Complète

Bienvenue dans ce projet complet d'implémentation de **modèles d'apprentissage automatique fondamentaux**. Ce repository explore les principaux algorithmes utilisés en science des données, avec des explications théoriques, des formules mathématiques et des implémentations pratiques.

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

**Concept :** Modéliser la relation linéaire entre une variable indépendante `X` et une variable dépendante `Y`.

**Équation :**
```
ŷ = β₀ + β₁x
```

Où :
- `β₀` : intercept (ordonnée à l'origine)
- `β₁` : coefficient directeur (pente)
- `ŷ` : valeur prédite

**Fonction de Coût (MSE - Mean Squared Error) :**
```
J(β₀, β₁) = (1/2m) Σᵢ₌₁ᵐ (hβ(xⁱ) - yⁱ)²
```

**Optimisation (Descente de Gradient) :**
```
β₀ := β₀ - α(1/m) Σᵢ₌₁ᵐ (hβ(xⁱ) - yⁱ)
β₁ := β₁ - α(1/m) Σᵢ₌₁ᵐ (hβ(xⁱ) - yⁱ)xⁱ
```

**Cas d'usage :**
- ✅ Prédire des prix (immobilier, actions)
- ✅ Analyser des tendances
- ✅ Relation linéaire claire entre variables

**Avantages & Inconvénients :**
| Avantages | Inconvénients |
|-----------|---------------|
| Rapide et simple | Ne capture que les relations linéaires |
| Interprétable | Sensible aux valeurs aberrantes |
| Calculs efficaces | Assume l'indépendance des erreurs |

---

### 1.2 Régression Linéaire Multiple

**Concept :** Généralisation de la régression simple avec `p` variables indépendantes.

**Équation matricielle :**
```
ŷ = Xβ
```

Où :
- `X` : matrice de design (n × p)
- `β` : vecteur de coefficients
- `ŷ` : vecteur de prédictions

**Solution analytique (Équations Normales) :**
```
β = (Xᵀ X)⁻¹ Xᵀ y
```

**Hypothèses du modèle (Gauss-Markov) :**
1. Linéarité : E[y|X] = Xβ
2. Pas de multicolinéarité : rang(X) = p
3. Homoscédasticité : Var(ε) = σ² I
4. Indépendance des erreurs
5. Normalité des erreurs : ε ~ N(0, σ²I)

**Métriques d'évaluation :**
```
R² = 1 - (SS_res / SS_tot)
    = 1 - Σ(yⁱ - ŷⁱ)² / Σ(yⁱ - ȳ)²

RMSE = √(1/m Σᵢ₌₁ᵐ (yⁱ - ŷⁱ)²)

MAE = 1/m Σᵢ₌₁ᵐ |yⁱ - ŷⁱ|
```

---

### 1.3 Régression Polynomiale

**Concept :** Modéliser des relations non-linéaires en créant des features polynomiales.

**Équation :**
```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βₚxᵖ
```

**Exemple : Polynomial d'ordre 3**
```
ŷ = β₀ + β₁x + β₂x² + β₃x³
```

**Transformation des features :**
```
Φ(x) = [1, x, x², x³, ..., xᵖ]
```

**Complexité & Risque de Surapprentissage :**
```
Erreur totale = Biais² + Variance + Bruit
- Ordre faible → Biais élevé (sous-apprentissage)
- Ordre élevé → Variance élevée (surapprentissage)
```

**Régularisation pour éviter le surapprentissage :**
```
J(β) = 1/2m Σᵢ₌₁ᵐ (hβ(xⁱ) - yⁱ)² + λ/2m Σⱼ₌₁ᵖ βⱼ²  (Ridge)
```

---

### 1.4 Régression Logistique

**Concept :** Classification binaire en modélisant la probabilité d'appartenance à la classe positive.

**Fonction Sigmoïde :**
```
σ(z) = 1 / (1 + e^(-z))

Où z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

**Hypothèse (Probabilité Prédite) :**
```
P(y=1|x) = σ(Xβ) = 1 / (1 + e^(-Xβ))
P(y=0|x) = 1 - σ(Xβ)
```

**Fonction de Coût (Log Loss / Cross-Entropy) :**
```
J(β) = -1/m Σᵢ₌₁ᵐ [yⁱ log(hβ(xⁱ)) + (1-yⁱ) log(1-hβ(xⁱ))]
```

**Interprétation des Coefficients :**
```
Pour un coefficient β_j :
- Une augmentation d'une unité en x_j 
  → Multiplicateur de odds = e^(β_j)
  → Changement en probabilité ≈ β_j/4 (quand P ≈ 0.5)
```

**Décision :**
```
Prédiction = 1 si P(y=1|x) ≥ 0.5
Prédiction = 0 si P(y=1|x) < 0.5
```

**Matrice de Confusion & Métriques :**
```
              Prédiction Positive | Prédiction Négative
Réalité Positive      TP (True +)  |      FN (Faux -)
Réalité Négative      FP (Faux +)  |      TN (True -)

Précision = TP / (TP + FP)           [Exactitude des prédictions positives]
Rappel    = TP / (TP + FN)           [Couverture des vrais positifs]
F1-Score  = 2(Précision × Rappel) / (Précision + Rappel)
AUC-ROC   = Aire sous la courbe ROC
```

---

## 🌳 2. Modèles d'Arbres

### 2.1 Arbres de Décision

**Concept :** Modèle hiérarchique qui divise l'espace des features selon des règles simples en cascade.

**Structure :**
```
                    X₁ ≤ 5?
                   /        \
                YES          NO
               /              \
          X₂ ≤ 10?          X₃ ≤ 20?
          /      \           /      \
        ...      ...       ...      ...
        
      [Feuilles = Classes/Valeurs prédites]
```

**Critères de Division (Split) :**

**Pour la Classification (Gini Index) :**
```
Gini(t) = 1 - Σⱼ₌₁ᶜ (pⱼ)²

Où pⱼ = proportion de classe j au nœud t

Gini_split = (n_left/n) × Gini(left) + (n_right/n) × Gini(right)

Gain = Gini(parent) - Gini_split
```

**Pour la Régression (Réduction de Variance) :**
```
Variance(t) = Σᵢ₌₁ⁿ (yⁱ - ȳ)² / n

Variance_reduction = Var(parent) - [(n_left/n) × Var(left) + (n_right/n) × Var(right)]
```

**Entropie Shannon :**
```
Entropie(t) = -Σⱼ₌₁ᶜ pⱼ log₂(pⱼ)

Information_Gain = Entropie(parent) - Entropie_split
```

**Avantages :**
- ✅ Interprétable et visualisable
- ✅ Pas de normalisation des données nécessaire
- ✅ Capture les non-linéarités
- ✅ Gère automatiquement les interactions

**Inconvénients :**
- ❌ Tendance au surapprentissage
- ❌ Instabilité (petits changements → gros changements dans l'arbre)
- ❌ Biais vers les features avec plus de valeurs
- ❌ Performances modérées sur données complexes

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

```
1. Bootstrap Sampling (Bagging)
   - Créer m échantillons bootstrap (avec remplacement)
   - Chaque bootstrap ≈ 63.2% des données originales
   
2. Entraîner un arbre sur chaque bootstrap
   - Avec randomisation des features à chaque split
   
3. Prédictions d'ensemble
   Classification  : Vote majoritaire
   Régression      : Moyenne des prédictions
```

**Formule de Prédiction :**

**Classification (Voting) :**
```
Ŷ = argmax_k Σᵢ₌₁ᵐ 𝕀(Tᵢ(x) = k)
```

**Régression (Averaging) :**
```
Ŷ = 1/m Σᵢ₌₁ᵐ Tᵢ(x)
```

**Réduction de Variance par Ensemble :**
```
Var(Ŷ_ensemble) = ρ·Var(Tree) + (1-ρ)/m · Var(Tree)

Où ρ = corrélation moyenne entre arbres

Idéalement ρ → 0 pour maximum de réduction
```

**Importance des Features (Mean Decrease in Impurity) :**
```
Importanceⱼ = 1/m Σᵢ₌₁ᵐ (Gini_before - Gini_after) × n_samples / n_total

où j est la j-ème feature
```

**Avantages :**
- ✅ Performance souvent meilleure que arbres individuels
- ✅ Réduit le surapprentissage (via diversité)
- ✅ Robuste aux valeurs aberrantes
- ✅ Traite bien les données déséquilibrées
- ✅ Calcule l'importance des features

**Hyperparamètres :**
| Paramètre | Recommandation |
|-----------|----------------|
| `n_estimators` | 100-1000 (plus = mieux, mais coûteux) |
| `max_depth` | None ou log₂(n_features) |
| `max_features` | √p (classification) ou p/3 (régression) |
| `min_samples_split` | 2-5 |
| `bootstrap` | True (obligatoire pour bagging) |

---

## 👥 3. Modèles de Voisinages (KNN)

### K-Nearest Neighbors

**Concept :** Classification/Régression basée sur les k voisins les plus proches dans l'espace des features.

**Algorithme :**
```
Pour une nouvelle observation x :
1. Calculer la distance entre x et tous les points d'entraînement
2. Sélectionner les k points les plus proches
3. Classification  : Vote majoritaire parmi les k voisins
   Régression      : Moyenne des valeurs des k voisins
```

**Fonctions de Distance :**

**Euclidienne :**
```
d(x, xⁱ) = √(Σⱼ₌₁ᵖ (xⱼ - xⱼⁱ)²)
```

**Manhattan :**
```
d(x, xⁱ) = Σⱼ₌₁ᵖ |xⱼ - xⱼⁱ|
```

**Minkowski :**
```
d(x, xⁱ) = (Σⱼ₌₁ᵖ |xⱼ - xⱼⁱ|^r)^(1/r)
```

**Cosinus :**
```
d(x, xⁱ) = 1 - (x · xⁱ) / (||x|| × ||xⁱ||)
```

**Prédiction (Classification avec poids) :**
```
P(y=c|x) = Σᵢ₌₁ᵏ wᵢ × 𝕀(yⁱ = c) / Σᵢ₌₁ᵏ wᵢ

Où wᵢ = 1/dᵢ (inverse de la distance)
```

**Impact du paramètre k :**

```
k petit (k=1)
├─ Modèle très local, fluctuant
├─ Biais faible, Variance élevée
└─ Risque de surapprentissage

k optimal
├─ Bon compromis biais-variance
├─ Généralement k = √n ou k ∈ [3,10]
└─ À valider par validation croisée

k grand (k=n)
├─ Prédiction = classe dominante
├─ Biais élevé, Variance faible
└─ Risque de sous-apprentissage
```

**Complexité Computationnelle :**
```
Entraînement : O(1)           [Pas d'entraînement réel]
Prédiction  : O(n × p × k)   [Calcul de distances + tri]
Espace      : O(n × p)       [Stockage de tout l'entraînement]

→ KNN est "Lazy Learner" : travail différé à la prédiction
```

**Prétraitements Essentiels :**
1. **Normalisation des features :**
   ```
   x_normalized = (x - mean) / std  [StandardScaler]
   ou
   x_normalized = (x - min) / (max - min)  [MinMaxScaler]
   ```
   ⚠️ Critique car KNN est basé sur les distances

2. **Réduction de dimensionnalité :**
   - PCA pour réduire la "malédiction de la dimensionnalité"
   - Sélection des features pertinentes

**Avantages :**
- ✅ Très simple à comprendre et implémenter
- ✅ Pas de phase d'entraînement
- ✅ Bon pour les données non-linéaires
- ✅ Pas d'hypothèses sur la distribution

**Inconvénients :**
- ❌ Lent en prédiction (calcul de toutes les distances)
- ❌ Sensible à l'ordre des features
- ❌ Performance dégradée en haute dimension (malédiction)
- ❌ Sensible aux valeurs aberrantes
- ❌ Gestion difficile des features catégoriques

---

## 🎲 4. Naive Bayes

### Gaussian Naive Bayes

**Concept :** Classifier probabiliste basé sur le théorème de Bayes avec l'hypothèse d'indépendance conditionnelle des features.

**Théorème de Bayes :**
```
P(y|X) = P(X|y) × P(y) / P(X)

Prédiction : ŷ = argmax_y P(y|X)
```

**Simplification - Hypothèse Naive (Indépendance) :**
```
P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₚ|y)

Donc : P(y|X) ∝ P(y) × ∏ⱼ₌₁ᵖ P(xⱼ|y)
```

**Gaussian Naive Bayes - Hypothèse de Normalité :**
```
P(xⱼ|y) ~ N(μⱼ,y, σⱼ,y²)

P(xⱼ|y) = 1/(√(2π σⱼ,y²)) × exp(-(xⱼ - μⱼ,y)²/(2σⱼ,y²))
```

**Entraînement :**
```
Pour chaque classe y et feature j :

μⱼ,y = 1/nᵧ Σᵢ: yⁱ=y xⱼⁱ              [Moyenne]

σⱼ,y² = 1/nᵧ Σᵢ: yⁱ=y (xⱼⁱ - μⱼ,y)²  [Variance]

P(y) = nᵧ / n                          [Probabilité a priori]
```

**Prédiction Logarithmique (pour stabilité numérique) :**
```
log P(y|X) ∝ log P(y) + Σⱼ₌₁ᵖ log P(xⱼ|y)

ŷ = argmax_y [log P(y) + Σⱼ₌₁ᵖ log(1/(√(2π σⱼ,y²))) - (xⱼ - μⱼ,y)²/(2σⱼ,y²)]
```

**Visualisation : Probabilités de Classes**

```
Données binaires, 2 features :

y=0 (Classe 0)          y=1 (Classe 1)
   X₂                      X₂
   |    ●  ●               |        ◯  ◯
   |  ●      ●             |    ◯      ◯
   |___________X₁          |___________X₁

P(y=0|X) = P(X|0)P(0)/P(X)
P(y=1|X) = P(X|1)P(1)/P(X)
```

**Avantages :**
- ✅ Très rapide, même sur larges datasets
- ✅ Fonctionne bien avec peu de données
- ✅ Interprétable (probabilités explicites)
- ✅ Bon pour textes et données éparses
- ✅ Gère bien les données manquantes

**Inconvénients :**
- ❌ Hypothèse d'indépendance rarement vraie
- ❌ Features fortement corrélées → problèmes
- ❌ Probabilité de prédiction biaisée
- ❌ Performance modérée sur relations complexes

**Cas d'Usage :**
- 📧 Filtrage de spam
- 📝 Classification de textes
- 🏥 Diagnostic médical
- ⭐ Systèmes de recommandation

---

## 🚀 5. Support Vector Machines (SVM)

**Concept :** Trouver l'hyperplan optimal qui maximise la marge entre les classes.

### 5.1 SVM Linéaire

**Problème d'Optimisation :**
```
Maximiser la marge = 2/||w||

Sous contrainte : yⁱ(wᵀxⁱ + b) ≥ 1  pour tout i

Formulation duale :
Minimiser : 1/2 ||w||² + C × Σᵢ₌₁ⁿ ξᵢ

Où ξᵢ = slack variables (tolérance de violation)
      C = paramètre de régularisation
```

**Hyperplan de Séparation :**
```
Hyperplan : wᵀx + b = 0
Distance d'un point à l'hyperplan : |wᵀxⁱ + b| / ||w||
Marge : 2 / ||w||
```

**Décision :**
```
ŷ = sign(wᵀx + b) = { +1 si wᵀx + b ≥ 0
                      -1 si wᵀx + b < 0
```

**Visualisation (Cas 2D) :**
```
      y=+1                    Hyperplan optimal
       ●    ....              (maximise marge)
            .  .
       ●   .    .   Marge     w ⊥ hyperplan
          .      .
    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     Vecteurs supports
         .      .
        .    .   ○
            .  .
           ○   y=-1
```

### 5.2 SVM Non-Linéaire (Kernel Trick)

**Problème :** Les données ne sont pas toujours linéairement séparables.

**Solution :** Transformer l'espace via une fonction φ(x) en espace de dimension supérieure.

**Kernel Trick :**
```
Au lieu de calculer φ(x) explicitement,
on utilise une fonction kernel : K(xⁱ, xʲ) = φ(xⁱ)ᵀ φ(xʲ)

Avantage : Calcul efficace sans connaître φ explicitement
```

**Kernels Courants :**

**1. Kernel Polynomial :**
```
K(x, x') = (γ x·x' + r)^d

Paramètres :
- γ : coefficient (généralement 1/p)
- d : degré du polynôme
- r : constante (décalage)

Exemple : d=3 (cubique)
K(x, x') = (x·x' + 1)³
```

**2. RBF (Radial Basis Function) - Gaussian :**
```
K(x, x') = exp(-γ ||x - x'||²)

γ = 1/(2σ²)  où σ est l'écart-type

Interprétation : Similarité locale autour de x
- γ petit   → Support global (décision lisse)
- γ grand   → Support local (décision complexe)
```

**3. Kernel Sigmoïde :**
```
K(x, x') = tanh(γ x·x' + r)
```

**Transformation d'Espace :**
```
Espace original (2D, non-linéaire)    Espace transformé (3D, linéaire)
                                      
    ●  ○  ○                          ●  ●  ●
   ●  ○  ●  ○                   →   ○  ○  ○
  ●  ○  ○  ●  ○                     ○  ○  ○

Non séparable linéairement          Linéairement séparable
```

### 5.3 SVM Multi-classe

**Stratégies :**

**One-vs-Rest :**
```
Pour K classes :
- Entraîner K modèles SVM binaires
- Chaque SVM sépare classe k du reste
- Prédiction : argmax_k score_k(x)
```

**One-vs-One :**
```
Pour K classes :
- Entraîner K(K-1)/2 modèles (paires)
- Prédiction : Classe avec plus de votes
```

### 5.4 Hyperparamètres Clés

| Paramètre | Impact | Recommandation |
|-----------|--------|----------------|
| `C` | Régularisation (inverse) | 0.1 à 100 (log scale) |
| `gamma` | Portée du kernel | 0.001 à 1 |
| `kernel` | Fonction de transformation | 'rbf' ou 'poly' |
| `degree` | Degré polynomial | 2 ou 3 |
| `class_weight` | Pondération des classes | 'balanced' si déséquilibre |

**Effet de C :**
```
C petit (C → 0)
├─ Marge large, erreurs tolérées
├─ Modèle simple, generalise bien
└─ Biais ↑, Variance ↓

C optimal
├─ Bon compromis
└─ À valider par validation croisée

C grand (C → ∞)
├─ Marge petite, peu d'erreurs tolérées
├─ Modèle complexe, surapprentissage risqué
└─ Biais ↓, Variance ↑
```

**Avantages :**
- ✅ Performant en haute dimension
- ✅ Utilise peu de ressources en prédiction (vecteurs supports)
- ✅ Flexible via kernels
- ✅ Théorie mathématique robuste
- ✅ Bon pour classification complexe

**Inconvénients :**
- ❌ Entraînement lent sur gros datasets (O(n²) ou O(n³))
- ❌ Moins interprétable que arbres/linéaire
- ❌ Normalisation des données essentielle
- ❌ Choix du kernel critique
- ❌ Gestion des multi-classes moins directe

---

## 📁 Structure du Projet

```
MODELES-DE-MACHINE-LEARNING/
│
├── README.md                           # Documentation principale (ce fichier)
│
├── Modeles Lineaires/                  # Régression & Classification linéaires
│   ├── Regression_Simple.py           # Régression linéaire simple
│   ├── Regression_Multiple.py         # Régression avec plusieurs variables
│   ├── Regression_Polynomiale.py      # Régressions polynomiales (d=2,3,4...)
│   └── Regression_Logistique.py       # Classification binaire
│
├── Modeles d'Arbres/                   # Arbres de décision & Random Forest
│   ├── Arbre_Decision_Classification.py
│   ├── Arbre_Decision_Regression.py
│   ├── Random_Forest_Classification.py
│   └── Random_Forest_Regression.py
│
├── Modeles de Voisinages/              # K-Nearest Neighbors
│   ├── KNN_Classification.py
│   └── KNN_Regression.py
│
├── Naive Bayes/                        # Classifieurs Bayésiens
│   └── Gaussian_Naive_Bayes.py
│
└── Support Vectors Machines/           # SVM
    ├── SVM_Classification_Linear.py
    ├── SVM_Classification_RBF.py
    ├── SVM_Classification_Polynomial.py
    └── SVM_Regression.py
```

---

## 🛠️ Installation & Utilisation

### Prérequis
```bash
Python >= 3.7
pip >= 20.0
```

### Installation des Dépendances
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Exemple d'Utilisation : Régression Linéaire

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Générer données
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2.5 * X.flatten() + 5 + np.random.normal(0, 2, 100)

# Entraînement / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"β₀ (intercept): {model.intercept_:.2f}")
print(f"β₁ (coefficient): {model.coef_[0]:.2f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Visualisation
plt.scatter(X_test, y_test, alpha=0.5, label='Données réelles')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Régression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Exemple : Classification avec Random Forest

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Données
iris = load_iris()
X, y = iris.data, iris.target

# Modèle
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Validation croisée
scores = cross_val_score(rf, X, y, cv=5)
print(f"Scores CV: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Entraînement complet
rf.fit(X, y)

# Rapport de classification
y_pred = rf.predict(X)
print(classification_report(y, y_pred, target_names=iris.target_names))

# Importance des features
for name, importance in zip(iris.feature_names, rf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

---

## 📚 Ressources & Références

### Livres
- **An Introduction to Statistical Learning** - James, Witten, Hastie, Tibshirani
- **The Elements of Statistical Learning** - Hastie, Tibshirani, Friedman
- **Pattern Recognition and Machine Learning** - Bishop
- **Machine Learning: A Probabilistic Perspective** - Murphy

### Cours
- Andrew Ng's Machine Learning (Coursera)
- Fast.ai - Practical Deep Learning
- MIT OpenCourseWare - Machine Learning

### Documentation
- [Scikit-learn Official Docs](https://scikit-learn.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## 🎓 Conseils d'Apprentissage

### Ordre Recommandé
1. **Débuter** : Régression Linéaire Simple → Multiple → Logistique
2. **Progresser** : Arbres de Décision → Random Forest
3. **Approfondissement** : KNN → Naive Bayes → SVM
4. **Combinaison** : Ensembles (Stacking, Blending)

### Bonnes Pratiques
✅ **Toujours explorer les données** (EDA)
✅ **Faire train/validation/test split**
✅ **Normaliser les features** (StandardScaler)
✅ **Utiliser validation croisée** (stratifiée pour déséquilibre)
✅ **Hypertune les hyperparamètres** (GridSearchCV, RandomizedSearchCV)
✅ **Vérifier les hypothèses du modèle**
✅ **Documenter les résultats**

### Pièges Courants
❌ **Data Leakage** : Information test fuit dans l'entraînement
❌ **Classe Imbalancée** : Sans stratification ni pondération
❌ **Pas de baseline** : Comparaison nécessaire
❌ **Overfitting silent** : Bon score train, mauvais score test
❌ **Features interactions** : Manquées par modèles trop simples

---

## 🚀 Prochaines Étapes

- [ ] Ajouter exemples complets (datasets réels)
- [ ] Visualisations interactives
- [ ] Comparaison entre modèles
- [ ] Techniques d'ensemble avancées
- [ ] Deep Learning basics
- [ ] Techniques de traitement du texte (NLP)
- [ ] Réduction de dimensionnalité (PCA, t-SNE)

---

## 📧 Contact & Contribution

Ce projet est une ressource pédagogique. N'hésitez pas à :
- 🐛 Signaler des bugs
- 💡 Proposer des améliorations
- 📝 Ajouter des explications
- 🔧 Enrichir les implémentations

---

**Dernière mise à jour :** 2026-09-11  
**Version :** 2.0  
**Licence :** MIT

---

## 📊 Tableau Comparatif - Choisir le Bon Modèle

| Critère | Linéaire | Arbre | RF | KNN | Naive B. | SVM |
|---------|----------|-------|-----|-----|----------|-----|
| **Vitesse entraînement** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Vitesse prédiction** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Interprétabilité** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Non-linéarité** | ❌ | ✅ | ✅ | ✅ | Limité | ✅ |
| **Données manquantes** | Nécessite traitement | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Scalabilité (gros n)** | ✅ | Modéré | Modéré | ❌ | ✅ | ❌ |
| **Haute dimension** | ✅ | Modéré | Modéré | ❌ | ✅ | ✅ |

---

*Bonne chance dans votre apprentissage du Machine Learning ! 🚀*
