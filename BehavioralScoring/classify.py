#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    classification-related classes and functions
"""

from __future__ import print_function

import os
import os.path as op
import inspect
import numpy as np
from sklearn import discriminant_analysis
from sklearn import svm
from sklearn import neural_network
from sklearn import cross_validation
from sklearn import grid_search


def get_sklearn_minor_version():

    import sklearn
    v = sklearn.__version__
    _, minor, _ = v.split('.')

    return int(minor)


_minor_version = get_sklearn_minor_version()
if _minor_version <= 16:
    _CLASS_WEIGHT_BALANCED = 'auto'
else:
    _CLASS_WEIGHT_BALANCED = 'balanced'


# -----------------------------------------------------------------------------
# some helpers
# -----------------------------------------------------------------------------

def makedirs_save(d):

    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except BaseException:
            pass


def run_job(data_path=None, param_index=None, fold=None):

    print("loading data from file", data_path)
    data = np.load(data_path)

    X = data['X']
    y = data['y']
    train_ind = data['train_ind'][fold]
    test_ind = data['test_ind'][fold]
    C = data['C_values'][param_index]

    clf = train_single_model(X[train_ind, :], y[train_ind], C)
    score = clf.score(X[test_ind, :], y[test_ind])

    file_name = 'cv_fold_%d_param_%d.npz' % (fold, param_index)
    result_file = op.join(op.split(data_path)[0], file_name)
    np.savez(result_file, param_index=param_index, fold=fold,
             score=score, classifier=clf)


def train_single_model(X, y, C, verbose=1):

    minor_version = get_sklearn_minor_version()
    if minor_version <= 16:
        class_weight = 'auto'
    else:
        class_weight = 'balanced'

    clf = svm.LinearSVC(verbose=verbose, C=C, class_weight=class_weight,
                        dual=False, penalty='l2', max_iter=1000)
    clf.fit(X, y)

    return clf


def compute_error_rates(y, y_pred, labels):

    n_labels = len(labels)
    tp_rate = np.zeros((n_labels,))
    fp_rate = np.zeros((n_labels))

    for i, label in enumerate(labels):

        v = y == label
        tpr = np.sum(y[v] == y_pred[v]) / float(np.sum(v))

        fp = np.sum(np.logical_and(y_pred == label, y != label))
        tn = np.sum(np.logical_and(y != label, y_pred != label))
        fpr = fp / float(fp + tn)

        tp_rate[i] = tpr
        fp_rate[i] = fpr

    return tp_rate, fp_rate


# -----------------------------------------------------------------------------
# Classifiers
# -----------------------------------------------------------------------------


class LDA(discriminant_analysis.LinearDiscriminantAnalysis):

    def __init__(self, **kwargs):
        super(LDA, self).__init__(**kwargs)

    def fit(self, X, y, **kwargs):

        labels = np.unique(y)
        priors = np.zeros((labels.shape[0],))
        for i, label in enumerate(labels):
            n = np.sum(y == label)
            priors[i] = n / float(y.shape[0])

        self.priors = priors

        super(LDA, self).fit(X, y)


class LinearSVM(svm.LinearSVC):

    def __init__(self, n_folds=4, C_values=None, n_jobs=-1,
                 class_weight=_CLASS_WEIGHT_BALANCED, **kwargs):

        # due to sklearn's way of using base classes for different MLPs
        # in combination with joblib which is used for cross-validation
        # we have to check what is being passed to the superclass' constructor
        sup = super(LinearSVM, self)
        argspecs = inspect.getargspec(sup.__init__)
        valid_kwargs = {}
        other_args = {}
        for k in kwargs:
            if k in argspecs.args:
                valid_kwargs[k] = kwargs[k]
            else:
                other_args[k] = kwargs[k]

        sup.__init__(**valid_kwargs)

        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.C_values = C_values

        self.__dict__.update(**other_args)

    def get_params(self, deep=True):

        return self.__dict__

    def fit(self, X, y, optimize_hyperparameters=False):

        if optimize_hyperparameters:

            cv = cross_validation.StratifiedKFold(y, n_folds=self.n_folds,
                                                  shuffle=True,
                                                  random_state=0)

            if self.C_values is None:
                C_values = 2. ** np.linspace(-10, 3, 20)
            else:
                C_values = self.C_values
            param_grid = {'C': C_values}
            grid = grid_search.GridSearchCV(self, param_grid,
                                            n_jobs=self.n_jobs,
                                            iid=True, refit=False,
                                            cv=cv, verbose=1)
            grid.fit(X, y)

            self.C = grid.best_params_['C']
            self.fit(X, y)

        else:
            print("Fitting model using C = ", self.C)
            super(LinearSVM, self).fit(X, y)


class RBFSVM(svm.SVC):
    """SVM with RBF kernel (very slow for large data sets)"""

    def __init__(self, n_folds=4, C_values=None, gamma_values='auto',
                 n_jobs=4, class_weight=_CLASS_WEIGHT_BALANCED, **kwargs):

        super(RBFSVM, self).__init__(class_weight=class_weight, **kwargs)

        self.n_folds = n_folds
        self.n_jobs = n_jobs

        self.C_values = C_values
        self.gamma_values = gamma_values

    def fit(self, X, y, optimize_hyperparameters=False):

        if optimize_hyperparameters:

            cv = cross_validation.StratifiedKFold(y, n_folds=self.n_folds,
                                                  shuffle=True,
                                                  random_state=0)

            if self.C_values is None:
                C_values = 2.**np.linspace(-8, 2, 10)
            else:
                C_values = self.C_values
            param_grid = {'C': C_values}

            if self.gamma_values == 'auto':
                self.gamma = 'auto'
            else:
                if self.gamma_values is None:
                    gamma_values = 2. ** np.linspace(-10, 2, 10)
                else:
                    gamma_values = self.gamma_values
                param_grid['gamma'] = gamma_values

            grid = grid_search.GridSearchCV(self, param_grid,
                                            n_jobs=self.n_jobs,
                                            iid=True, refit=True,
                                            cv=cv, verbose=1)
            grid.fit(X, y)

        else:
            super(RBFSVM, self).fit(X, y)


class MLP(neural_network.MLPClassifier):
    """MLP for classification (optimized using ADAM)"""

    def __init__(self, alpha_values=None, n_jobs=-1, n_folds=4,
                 **kwargs):

        # due to sklearn's way of using base classes for different MLPs
        # in combination with joblib which is used for cross-validation
        # we have to check what is being passed to the superclass' constructor
        sup = super(MLP, self)
        argspecs = inspect.getargspec(sup.__init__)
        valid_kwargs = {}
        other_args = {}
        for k in kwargs:
            if k in argspecs.args:
                valid_kwargs[k] = kwargs[k]
            else:
                other_args[k] = kwargs[k]

        sup.__init__(**valid_kwargs)

        self.alpha_values = alpha_values
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.__dict__.update(**other_args)

    def get_params(self, deep=True):

        return self.__dict__

    def fit(self, X, y, optimize_hyperparameters=False):

        if optimize_hyperparameters:

            cv = cross_validation.StratifiedKFold(y, n_folds=self.n_folds,
                                                  shuffle=True,
                                                  random_state=0)

            if self.alpha_values is None:
                alpha_values = 2. ** np.linspace(-5, 5, 10)
            else:
                alpha_values = self.alpha_values

            param_grid = {'alpha': alpha_values}

            grid = grid_search.GridSearchCV(self, param_grid,
                                            n_jobs=self.n_jobs,
                                            iid=True, refit=False,
                                            cv=cv, verbose=1)
            grid.fit(X, y)

            self.alpha = grid.best_params_['alpha']
            self.fit(X, y)

        else:
            print("Fitting model using alpha = ", self.alpha)
            super(MLP, self).fit(X, y)
