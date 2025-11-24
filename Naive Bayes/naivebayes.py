import numpy as np

class GaussianNaiveBayes:

    def fit(self, X, y):
        nsamples, nfeatures = X.shape
        self._classes = np.unique(y)
        nclasses = len(self._classes)

        self._mean = np.zeros((nclasses, nfeatures), dtype=np.float64)
        self._var = np.zeros((nclasses, nfeatures), dtype=np.float64)
        self._priors = np.zeros(nclasses, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / nsamples

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        log_posteriors = []

        for idx, c in enumerate(self._classes):
            # log-prior
            log_prior = np.log(self._priors[idx])

            # somme des log-densités pour chaque feature
            log_likelihood = np.sum(self._log_gaussian(idx, x))

            log_posteriors.append(log_prior + log_likelihood)

        return self._classes[np.argmax(log_posteriors)]

    def _log_gaussian(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx]

        return -0.5 * np.log(2 * np.pi * var) - ((x - mean)**2 / (2 * var))
