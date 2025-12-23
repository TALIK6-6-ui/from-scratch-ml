# src/naive_bayes.py

import numpy as np

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None

    def fit(self, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)
        self.classes = np.unique(y)
        n_samples = len(y)

        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)
            self.class_priors[c] = np.log(n_c / n_samples)
            prob_1 = (np.sum(X_c, axis=0) + 1) / (n_c + 2)
            prob_0 = 1 - prob_1
            self.feature_probs[c] = {
                'log_prob_1': np.log(prob_1),
                'log_prob_0': np.log(prob_0)
            }

    def predict(self, X_test):
        X = np.array(X_test)
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                log_prob = self.class_priors[c]
                log_prob += np.sum(
                    x * self.feature_probs[c]['log_prob_1'] +
                    (1 - x) * self.feature_probs[c]['log_prob_0']
                )
                posteriors[c] = log_prob
            pred = max(posteriors, key=posteriors.get)
            predictions.append(pred)
        return np.array(predictions)


class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.word_probs = {}
        self.classes = None
        self.vocab_size = None

    def fit(self, X_train, y_train):
        X = np.array(X_train)
        y = np.array(y_train)
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]

        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)
            total_words = np.sum(X_c)
            self.class_priors[c] = np.log(n_c / len(y))
            word_counts = np.sum(X_c, axis=0)
            prob = (word_counts + 1) / (total_words + self.vocab_size)
            self.word_probs[c] = np.log(prob)

    def predict(self, X_test):
        X = np.array(X_test)
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                log_prior = self.class_priors[c]
                log_likelihood = np.dot(x, self.word_probs[c])
                posteriors[c] = log_prior + log_likelihood
            pred = max(posteriors, key=posteriors.get)
            predictions.append(pred)
        return np.array(predictions)
