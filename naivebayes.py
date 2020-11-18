#!/usr/bin/python3

import numpy as np
import pandas as pd

from abc import ABC, abstractclassmethod
from sklearn.model_selection import train_test_split

class naivebayes(ABC):
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

    @abstractclassmethod
    def fit(self, features, labels):
        pass

    @abstractclassmethod
    def predict(self, data):
        pass

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

class MultinomialNB(naivebayes):
    def __init__(self, priors=None):
        self.n_classes_ = None

        if priors is not None:
            self._check_priors(priors)

        self.prior_proba_ = priors

    def __str__(self):
        return "MultinomialNB(priors="+str(self.prior_proba_)+")"

    def fit(self, features, labels):
        self._check_dtype(features)
        self._check_dtype(labels)

        self.total_samples_ = len(labels)
        self.n_classes_ = np.unique(labels)
        self.class_hist_ = dict()
        self._class_len = dict()

        if self.prior_proba_ is None:
            self.prior_proba_ = dict()
        else:
            if len(self.prior_proba_) != self.n_classes_:
                raise ValueError("Number of classes should be", self.n_classes_)

        total_str = " ".join(features)
        for c in self.n_classes_:
            full_str = " ".join(features[labels == c])
            unique_words = len(set(full_str.split()))
            tmp = dict()

            self._class_len[c] = len(full_str.split())
            for word in total_str.split():
                tmp[word] = (full_str.split().count(word) + 1)/(self._class_len[c] + unique_words)
            self.class_hist_[c] = tmp.copy()

            if isinstance(self.prior_proba_, dict):
                self.prior_proba_[c] = np.sum((labels == c)) / self.total_samples_

        if labels.dtype == 'O':
            self.class_encoding_ = dict()
            for i, c in enumerate(self.n_classes_):
                self.class_encoding_[c] = i

        return

    def predict(self, data):
        self._check_dtype(data)

        pred = np.array([list(self.prior_proba_.values())] * len(data))
        for j, c in enumerate(self.n_classes_):
            for i, text in enumerate(data):
                for word in text.split():
                    pred[i, j] *= self.class_hist_[c].get(word, 1/self._class_len[c])

        pred = np.argmax(pred, axis=1)

        return pred

class GaussianNB(naivebayes):
    def __init__(self, priors=None):
        self.n_classes_ = None

        if priors is not None:
            self._check_priors(priors)

        self.prior_proba_ = priors

    def __str__(self):
        return "GaussianNB(priors="+str(self.prior_proba_)+")"

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

if __name__ == '__main__':
    data = pd.read_csv('imdb.zip', compression='zip')

    train_X, test_X, train_y, test_y = train_test_split(data.loc[:, data.columns != 'sentiment'], data['sentiment'], train_size=0.8)

    clf = MultinomialNB()
    clf.fit(np.squeeze(train_X.values, 1), train_y.values)

    print(clf.predict(np.squeeze(test_X.values, 1)))
    print("MSE on testing data", clf.evaluate(np.squeeze(test_X.values, 1), test_y.values, metrics='MSE'))