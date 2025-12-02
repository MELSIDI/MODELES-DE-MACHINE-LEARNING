import numpy as np

class SVM:

    def fit(self, X, y, alpha=0.001, _lambda=0.0, niters=1000, stop_crit=None):
        y_svm = np.where(y > 0, 1, -1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(niters):

            old_w = self.w.copy()

            for i in range(n_samples):

                marge = y_svm[i] * (np.dot(self.w, X[i]) - self.b)

                # --- Cas 1 : marge respectée => régularisation uniquement ---
                if marge >= 1:
                    self.w -= alpha * (2 * _lambda * self.w)
                # --- Cas 2 : marge violée => hinge loss + régularisation ---
                else:
                    self.w -= alpha * (2 * _lambda * self.w - y_svm[i] * X[i])
                    self.b += alpha * y_svm[i]

            # ---- Critère d'arrêt ----
            if stop_crit is not None:
                if np.linalg.norm(self.w - old_w) < stop_crit:
                    break


    def predict(self, X):
        linear = np.dot(X, self.w) - self.b
        return np.sign(linear)
