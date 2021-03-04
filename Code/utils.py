import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine, load_boston
from sklearn.utils import shuffle


def perturb(x, n_masked_features=None):
    if n_masked_features is None:
        # Gaussian noise
        scale = 1.0  # Scale/amount/variance of noise
        return x + np.random.normal(scale=scale, size=x.shape)
    else:
        # Feature masking (set some features to zero)
        idx = np.random.choice(range(x.shape[0]), size=n_masked_features, replace=False)
        return np.array([x[i] if i not in idx else 0. for i in range(x.shape[0])])


def compare_cf(xcf1, xcf2):
    return np.sum(np.abs(xcf1 - xcf2))


def load_data_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)

    return X, y


def load_data_wine():
    X, y = load_wine(return_X_y=True)

    return X, y


def load_data_digits():
    X, y = load_digits(return_X_y=True)

    return X, y
