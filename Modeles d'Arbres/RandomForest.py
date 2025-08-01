from DecisionTree import DecisionTreeClassifier
from DecisionTree import DecisionTreeRegressor
import numpy as np
from collections import Counter


class RandomForestClassifier:
    def __init__(self, min_samples_split: int = 2, max_depth: int = 10, n_tree=10, * ,criterion: str = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.n_tree = n_tree
        self.trees = []
        
        
    def fit(self, X, y):
        for _ in range(self.n_tree):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                          max_depth=self.max_depth,
                                          criterion=self.criterion)
            X_samples, y_samples = self.bootstrap(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree)
            
            
    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    
    def predict(self, X):
       # Collecte les prédictions de chaque arbre
       tree_preds = np.array([tree.predict(X) for tree in self.trees])
       # Effectue un vote majoritaire pour chaque exemple
       y_pred = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
       return np.array(y_pred)
   
    
    def score(self, y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)
    
    

class RandomForestRegressor:
    def __init__(self, min_samples_split: int = 2, max_depth: int = 10, n_tree=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_tree = n_tree
        self.trees = []
        
        
    def fit(self, X, y):
        for _ in range(self.n_tree):
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split,
                                          max_depth=self.max_depth)
            X_samples, y_samples = self.bootstrap(X, y)
            tree.fit(X_samples, y_samples)
            self.trees.append(tree)
            
            
    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    
    def predict(self, X):
       # Collecte les prédictions de chaque arbre
       tree_preds = np.array([tree.predict(X) for tree in self.trees])
       # Effectue un vote majoritaire pour chaque exemple
       return np.mean(tree_preds, axis=0)
   
    
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
            
    