#!/usr/bin/python3

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class GaussianNB():
    def __init__(self, priors=None):
        self.n_classes_ = None

        if priors is not None:
            self._check_priors(priors)

        self.prior_proba_ = priors

    def __str__(self):
        return "GaussianNB(priors="+str(self.prior_proba_)+")"

    def _check_dtype(self, arr):
        if not isinstance(arr, np.ndarray):
            raise ValueError("Expects only numpy ndarrays")

        return

    def _check_priors(self, priors):
        self._check_dtype(priors)

        if (priors < 0).any():
            raise ValueError("Probablities cannot be less than 0")

        if (priors > 1).any():
            raise ValueError("Probablities cannot be greater than 0")

        if int(sum(priors)) != 1:
            raise ValueError("The sum of the priors should be 1")

        return

    def fit(self, features, labels):
        self._check_dtype(features)
        self._check_dtype(labels)

        self.total_samples_ = len(labels)
        self.n_features_ = features.shape[1]
        self.n_classes_ = np.unique(labels)
        self.class_mean_ = dict()
        self.class_std_ = dict()

        if self.prior_proba_ is None:
            self.prior_proba_ = dict()
        else:
            if len(self.prior_proba_) != self.n_classes_:
                raise ValueError("Number of classes should be", self.n_classes_)

        for c in self.n_classes_:
            self.class_mean_[c] = np.mean(features[labels == c], axis=0)
            self.class_std_[c] = np.std(features[labels == c], axis=0)

            if isinstance(self.prior_proba_, dict):
                self.prior_proba_[c] = np.sum((labels == c)) / self.total_samples_

        if labels.dtype == 'O':
            self.class_encoding_ = dict()
            for i, c in enumerate(self.n_classes_):
                self.class_encoding_[c] = i

        return

    def predict(self, data):
        self._check_dtype(data)

        if self.n_features_ != data.shape[1]:
            raise ValueError("Number of features should be", self.n_features_)

        pred = np.zeros((data.shape[0], len(self.n_classes_)))
        for i, c in enumerate(self.n_classes_):
            term1 = (1 / (np.sqrt(2 * np.pi * self.class_std_[c]**2)))
            term2 = np.exp(-((data - self.class_mean_[c])**2 / (2 * self.class_std_[c]**2)))
            tmp = np.sum(np.log(self.prior_proba_[c] * term1 * term2), axis=1) 
            pred[:, i] = tmp

        pred = np.argmax(pred, axis=1)

        return pred

    def evaluate(self, X, y, metrics='R2'):
        self._check_dtype(X)
        self._check_dtype(y)

        if metrics not in ['R2', 'MSE', 'MAE']:
            raise ValueError("Possible metrics are [R2, MSE, MAE]")

        if y.dtype == 'O':
            y = np.vectorize(self.class_encoding_.get)(y)

        pred = self.predict(X)

        if metrics.lower() == 'mse':
            score = np.mean(np.subtract(y, pred)**2)
        elif metrics.lower() == 'mae':
            score = np.mean(np.abs(np.subtract(y, pred)))
        elif metrics.lower() == 'r2':
            score = 1 - ((np.sum(np.square(y - pred))) / (np.sum(np.square(y - np.mean(y)))))

        return score

if __name__ == '__main__':
    data = pd.read_csv('Iris.csv', index_col='Id')

    train_X, test_X, train_y, test_y = train_test_split(data.loc[:, data.columns != 'Species'], data['Species'], train_size=0.8)

    clf = GaussianNB()
    clf.fit(train_X.values, train_y.values)

    print(clf.predict(test_X.values))
    print("R^2 score on testing data", clf.evaluate(test_X.values, test_y.values))