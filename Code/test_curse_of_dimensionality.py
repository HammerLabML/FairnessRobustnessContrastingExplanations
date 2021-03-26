# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import cvxpy as cp
import random
import pandas as pd
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn_lvq import GlvqModel, GmlvqModel

from utils import compare_cf, perturb
from plausible_counterfactuals import LvqCounterfactual, MatrixLvqCounterfactual, FeasibleCounterfactualOfDecisionTree, FeasibleCounterfactualSoftmax


#modeldesc = "logreg"
modeldesc = "dectree"
#modeldesc = "glvq";n_prototypes=3

if __name__ == "__main__":
    features = [2, 4, 8, 16, 32, 64, 128]
    unfairness = []
    for n_features in features:
        n_kf_splits = 4
        X, y = make_blobs(n_samples=1000, centers=2, cluster_std=5., n_features=n_features)

        scores_cf_perturbation_dist = []
        results = {'notFound': 0, 'found': 0}

        kf = KFold(n_splits=n_kf_splits)
        for train_index, test_index in kf.split(X):
            # Split data into training and test set
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit and evaluate classifier
            model = None
            if modeldesc == "glvq":
                model = GlvqModel(prototypes_per_class=n_prototypes)
            elif modeldesc == "gmlvq":
                model = GmlvqModel(prototypes_per_class=n_prototypes)
            elif modeldesc == "logreg":
                model = LogisticRegression(multi_class='multinomial')
            elif modeldesc == "dectree":
                model = DecisionTreeClassifier(max_depth=7)
            model.fit(X_train, y_train)

            # Compute accuracy on test set
            y_pred = model.predict(X_test)
            print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")

            labels = np.unique(y)
            # Compute counterfactual of each test sample
            for i in range(X_test.shape[0]):
                x_orig_orig = X_test[i,:]
                y_orig = y_test[i]
                y_target = random.choice(list(filter(lambda l: l != y_test[i], labels)))

                cf = None
                if modeldesc == "logreg":
                    cf = FeasibleCounterfactualSoftmax(model.coef_, model.intercept_, X=X_train, ellipsoids_r=np.array([]), gmm_weights=np.array([0]), gmm_means=np.array([]), gmm_covariances=np.array([]), projection_matrix=None, projection_mean_sub=None)
                elif modeldesc == "glvq":
                    cf = LvqCounterfactual(model, X=X_train, ellipsoids_r=np.array([]), gmm_weights=np.array([0]), gmm_means=np.array([]), gmm_covariances=np.array([]), projection_matrix=None, projection_mean_sub=None)
                elif modeldesc == "gmlvq":
                    cf = MatrixLvqCounterfactual(model, X=X_train, ellipsoids_r=np.array([]), gmm_weights=np.array([0]), gmm_means=np.array([]), gmm_covariances=np.array([]), projection_matrix=None, projection_mean_sub=None)
                elif modeldesc == "dectree":
                    cf = FeasibleCounterfactualOfDecisionTree(model, X=X_train, ellipsoids_r=np.array([]), gmm_weights=np.array([0]), gmm_means=np.array([]), gmm_covariances=np.array([]), projection_matrix=None, projection_mean_sub=None)
                xcf = cf.compute_counterfactual(x_orig_orig, y_target=y_target, use_density_constraints=False)
                if xcf is None:
                    #print("No counterfactual found!")
                    results["notFound"] += 1
                    continue

                # Compute counterfactual of perturbed sample
                x_perturb = perturb(x_orig_orig)  # Perturb original data point
                x_perturb_t = [x_perturb]
                if model.predict(x_perturb_t) != y_orig:    
                    print("Perturbed sample is missclassified")

                x_perturbed_cf = cf.compute_counterfactual(x_perturb, y_target=y_target, use_density_constraints=False)
                if x_perturbed_cf is None:
                    #print("No counterfactual of perturbed sample found!")
                    results["notFound"] += 1
                    continue

                # Evaluate and store closeness
                results['found'] += 1
                cf_perturbation_dist = compare_cf(xcf, x_perturbed_cf)     # Distance between counterfactual of perturned and original sample
                scores_cf_perturbation_dist.append(cf_perturbation_dist)
        
        print(f"n_features={n_features}")
        print(f"Not found {results['notFound']}/{results['notFound'] + results['found']}")
        print("Without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_cf_perturbation_dist), np.mean(scores_cf_perturbation_dist), np.var(scores_cf_perturbation_dist)))
        unfairness.append([np.median(scores_cf_perturbation_dist), np.mean(scores_cf_perturbation_dist), np.var(scores_cf_perturbation_dist)])

    # Summary plot
    unfairness = np.array(unfairness)[:,0] # Select the median only!
    plt.plot(features, unfairness, 'o-', label="Median (Un)fairness")
    plt.xlabel("Number of features")
    plt.ylabel("Dist Cf of original vs. perturbed sample")
    plt.xticks(features)
    plt.legend()
    plt.show()
    #plt.savefig(f"exp_results_curseofdimensionality/{modeldesc}_perturbation_dist_curseofdimensionality.pdf", dpi=500, bbox_inches='tight', pad_inches = 0)