import numpy as np
from typing import Optional
from collections import Counter


class Node:
    """
    Représente un nœud dans un arbre de décision.

    Un nœud peut être :
    - un nœud interne : il contient un test sur une feature (feature_index < threshold)
    - une feuille : il contient une prédiction (classe ou valeur)

    Attributs :
    ----------
    left : Optional[Node]
        Sous-nœud gauche (échantillons où x[feature_index] < threshold).
    right : Optional[Node]
        Sous-nœud droit (échantillons où x[feature_index] >= threshold).
    feature_index : int
        Indice de la feature utilisée pour le split (None si feuille).
    threshold : float
        Seuil utilisé pour diviser les données (None si feuille).
    is_leaf : bool
        Indique si le nœud est une feuille (True) ou un nœud interne (False).
    value : Any
        Valeur prédite si le nœud est une feuille (ex : classe majoritaire ou moyenne).

    Méthodes :
    ---------
    predict(x):
        Prédit la sortie pour un échantillon `x` en descendant récursivement dans l'arbre.
    """
    def __init__(self, left: Optional['Node'] = None, right: Optional['Node'] = None, feature_index: int = None, threshold: float = None, is_leaf: bool = False,*,value=None):
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.value = value
        
    def predict(self, x):
        if self.is_leaf:
            return self.value
        if x[self.feature_index] < self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)
        
        
        

class DecisionTreeClassifier:
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, *, criterion: str = "entropy"):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        
        
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        
        
    def build_tree(self, X, y, *, depth: int = 0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            majority_class = Counter(y).most_common(1)[0][0]           
            return Node(is_leaf=True, value=majority_class)
        else:
            max_feature = X.shape[1]
            best_feature = None
            best_threshold = None
            best_splits = None
            best_information_gain = float('-inf')
            for feature in range(0, max_feature):
                thresholds = self._threshold(X[:, feature])
                for thr in thresholds:
                    left_idxs = np.where(X[:, feature] < thr)[0]
                    right_idxs = np.where(X[:, feature] >= thr)[0]
                    if len(left_idxs) == 0 or len(right_idxs) == 0:
                        continue
                    
                    y_left = y[left_idxs]
                    y_right = y[right_idxs]
                    impurity_parent = self.impurity(y)
                    if impurity_parent == 0:
                        return Node(is_leaf=True, value=y[0])
                    
                    impurity_left = self.impurity(y_left)
                    impurity_right = self.impurity(y_right)
                    weighted_avg = (len(y_left) / len(y)) * impurity_left + (len(y_right) / len(y)) * impurity_right
                    information_gain = impurity_parent - weighted_avg
                    if information_gain > best_information_gain:
                        best_information_gain = information_gain
                        best_feature = feature
                        best_threshold = thr
                        best_splits = (left_idxs, right_idxs)
            
            if best_information_gain == float('-inf'):
                majority_class = Counter(y).most_common(1)[0][0]
                return Node(is_leaf=True, value=majority_class)
             
            left = self.build_tree(X[best_splits[0]], y[best_splits[0]], depth=depth + 1)
            right = self.build_tree(X[best_splits[1]], y[best_splits[1]], depth=depth + 1)
            return Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)
        
                
    def _threshold(self, X):
        X_sorted = np.sort((np.unique(X)))
        max_i = len(X_sorted) - 1
        thresholds = [(X_sorted[i] + X_sorted[i + 1]) / 2 for i in range(max_i)]
        return thresholds
    
    
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return - np.sum(proportions * np.log2(proportions + 1e-9))
    
    
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)
    
    
    def impurity(self, y):
        if len(y) == 0:
            raise ValueError("Unsuported Range")
        if self.criterion == "entropy":
            return self.entropy(y)
        elif self.criterion == "gini":
            return self.gini(y)
        else:
            raise ValueError("Unsupported Criterion")
        
    
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])
    
    
    def score(self, y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)

                


class DecisionTreeRegressor:
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        
        
    def build_tree(self, X, y, *, depth: int = 0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            average_y = np.mean(y)           
            return Node(is_leaf=True, value=average_y)
        else:
            max_feature = X.shape[1]
            best_feature = None
            best_threshold = None
            best_splits = None
            best_information_gain = float('-inf')
            for feature in range(0, max_feature):
                thresholds = self._threshold(X[:, feature])
                for thr in thresholds:
                    left_idxs = np.where(X[:, feature] < thr)[0]
                    right_idxs = np.where(X[:, feature] >= thr)[0]
                    if len(left_idxs) == 0 or len(right_idxs) == 0:
                        continue
                    
                    y_left = y[left_idxs]
                    y_right = y[right_idxs]
                    impurity_parent = self.impurity(y)
                    if impurity_parent == 0:
                        return Node(is_leaf=True, value=y.mean())
                    
                    impurity_left = self.impurity(y_left)
                    impurity_right = self.impurity(y_right)
                    weighted_avg = (len(y_left) / len(y)) * impurity_left + (len(y_right) / len(y)) * impurity_right
                    information_gain = impurity_parent - weighted_avg
                    if information_gain > best_information_gain:
                        best_information_gain = information_gain
                        best_feature = feature
                        best_threshold = thr
                        best_splits = (left_idxs, right_idxs)
            
            if best_information_gain == float('-inf'):
                average_y = y.mean()
                return Node(is_leaf=True, value=average_y)
             
            left = self.build_tree(X[best_splits[0]], y[best_splits[0]], depth=depth + 1)
            right = self.build_tree(X[best_splits[1]], y[best_splits[1]], depth=depth + 1)
            return Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)
        
                
    def _threshold(self, X):
        X_sorted = np.sort((np.unique(X)))
        max_i = len(X_sorted) - 1
        thresholds = [(X_sorted[i] + X_sorted[i + 1]) / 2 for i in range(max_i)]
        return thresholds
    
    
    def impurity(self, y):
        if len(y) == 0:
            raise ValueError("Unsuported Range")
        return np.var(y)
        
    
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])
    
    
    def score(self, y_true, y_pred, *, metric='r2'):
        if metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'mse':
            mse = np.mean((y_true - y_pred) ** 2)
            return 1 / (1 + mse)
        else:
            raise ValueError("Unsuported metric")