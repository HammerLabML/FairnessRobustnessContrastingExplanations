# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import cvxpy as cp
import random
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn_lvq import GlvqModel, GmlvqModel

from plausible_counterfactuals import LvqCounterfactual, MatrixLvqCounterfactual, FeasibleCounterfactualOfDecisionTree, FeasibleCounterfactualSoftmax, HighDensityEllipsoids
from utils import compare_cf, perturb, load_data_breast_cancer, load_data_digits, load_data_wine


n_kf_splits = 4


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <dataset_desc> <model_desc>")
    else:
        datasetdesc = sys.argv[1]
        modeldesc = sys.argv[2]

        n_prototypes = 3

        # Load data
        if datasetdesc == "wine":
            X, y = load_data_wine();pca_dim = None
        elif datasetdesc == "breastcancer":
            X, y = load_data_breast_cancer();pca_dim = 5
        elif datasetdesc == "digits":
            X, y = load_data_digits();pca_dim = 40

        X, y = shuffle(X, y, random_state=42)

        labels = np.unique(y)

        # Global results for plots
        dist_perturbed_with_density_constraint = []
        dist_perturbed_without_density_constraint = []

        # Perturbations
        n_features = X.shape[1]     # Mask no (gaussian noise) up to half of all features
        masked_features = [None]
        masked_features += list(range(1, int(n_features / 2) + 1))
        for feature_mask in masked_features:
            # Results
            scores_with_density_constraint = []
            scores_without_density_constraint = []
            scores_perturbed_with_density_constraint = []
            scores_perturbed_without_density_constraint = []
            distances_with_density_constraint = []
            distances_without_density_constraint = []
            distances_perturbed_with_density_constraint = []
            distances_perturbed_without_density_constraint = []
            original_data = []
            original_data_labels = []
            cfs_with_density_constraint = []
            cfs_without_density_constraint = []
            cfs_perturbed_with_density_constraint = []
            cfs_perturbed_without_density_constraint = []
            cfs_target_label = []
            scores_cf_perturbation_dist = []
            scores_cf_feasible_perturbation_dist = []
            results = {'notFound': 0, 'found': 0}
            n_wrong_classification = 0

            kf = KFold(n_splits=n_kf_splits)
            for train_index, test_index in kf.split(X):
                # Split data into training and test set
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # If requested: Reduce dimensionality
                X_train_orig = np.copy(X_train)
                X_test_orig = np.copy(X_test)
                projection_matrix = None
                projection_mean_sub = None
                pca = None
                if pca_dim is not None:
                    pca = PCA(n_components=pca_dim)
                    pca.fit(X_train)

                    projection_matrix = pca.components_ # Projection matrix
                    projection_mean_sub = pca.mean_
                    #print(projection_matrix)

                    X_train = np.dot(X_train - projection_mean_sub, projection_matrix.T)
                    X_test = np.dot(X_test - projection_mean_sub, projection_matrix.T)
                
                # Fit classifier
                model = None
                if modeldesc == "glvq":
                    model = GlvqModel(prototypes_per_class=n_prototypes, random_state=4242)
                elif modeldesc == "gmlvq":
                    model = GmlvqModel(prototypes_per_class=n_prototypes, random_state=4242)
                elif modeldesc == "logreg":
                    model = LogisticRegression(multi_class='multinomial')
                elif modeldesc == "dectree":
                    model = DecisionTreeClassifier(max_depth=7, random_state=42)
                model.fit(X_train, y_train)

                # Compute accuracy on test set
                y_pred = model.predict(X_test)
                print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")

                # Fit model for finding closest samples
                closest_samples = ClosestSample(X_train_orig, y_train)

                # For each class, fit density estimators
                density_estimators = {}
                kernel_density_estimators = {}
                labels = np.unique(y)
                for label in labels:
                    # Get all samples with the 'correct' label
                    idx = y_train == label
                    X_ = X_train[idx, :]

                    # Optimize hyperparameters
                    cv = GridSearchCV(estimator=KernelDensity(), iid=False, param_grid={'bandwidth': np.arange(0.1, 10.0, 0.05)}, n_jobs=-1, cv=5)
                    cv.fit(X_)
                    bandwidth = cv.best_params_["bandwidth"]
                    print("bandwidth: {0}".format(bandwidth))

                    cv = GridSearchCV(estimator=GaussianMixture(covariance_type='full'), iid=False, param_grid={'n_components': range(2, 10)}, n_jobs=-1, cv=5)
                    cv.fit(X_)
                    n_components = cv.best_params_["n_components"]
                    print("n_components: {0}".format(n_components))

                    # Build density estimators
                    kde = KernelDensity(bandwidth=bandwidth)
                    kde.fit(X_)

                    de = GaussianMixture(n_components=n_components, covariance_type='full')
                    de.fit(X_)

                    density_estimators[label] = de
                    kernel_density_estimators[label] = kde
                
                # For each point in the test set, compute a closest and a plausible counterfactual
                n_test = X_test.shape[0]
                for i in range(n_test):
                    x_orig = X_test[i,:]
                    x_orig_orig = X_test_orig[i,:]
                    y_orig = y_test[i]

                    y_target = random.choice(list(filter(lambda l: l != y_test[i], labels)))

                    if(model.predict([x_orig]) == y_target):  # Model already predicts target label!
                        continue
                    if(model.predict([x_orig]) != y_orig):  # Data point is missclassified
                        print("Original sample is missclassified")
                        continue

                    # Compute counterfactual WITH kernel density constraints
                    idx = y_train == y_target
                    X_ = X_train[idx, :]

                    # Build density estimator
                    de = density_estimators[y_target]
                    kde = kernel_density_estimators[y_target]

                    from scipy.stats import multivariate_normal
                    densities_training_samples = []
                    densities_training_samples_ex = []
                    for j in range(X_.shape[0]):
                        x = X_[j,:]
                        z = []
                        dim = x.shape[0]
                        for i in range(de.weights_.shape[0]):
                            x_i = de.means_[i]
                            w_i = de.weights_[i]
                            cov = de.covariances_[i]
                            cov = np.linalg.inv(cov)

                            b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
                            z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b) # NLL
                        densities_training_samples.append(np.min(z))
                        densities_training_samples_ex.append(z)
                    densities_training_samples = np.array(densities_training_samples)
                    densities_training_samples_ex = np.array(densities_training_samples_ex)

                    # Compute soft cluster assignments
                    cluster_prob_ = de.predict_proba(X_)
                    X_densities = de.score_samples(X_)
                    density_threshold = np.median(densities_training_samples)
                    r = HighDensityEllipsoids(X_, densities_training_samples_ex, cluster_prob_, de.means_, de.covariances_, density_threshold).compute_ellipsoids()

                    # Compute counterfactual
                    cf = None
                    if modeldesc == "glvq":
                        cf = LvqCounterfactual(model, X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
                    elif modeldesc == "gmlvq":
                        cf = MatrixLvqCounterfactual(model, X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
                    elif modeldesc == "logreg":
                        cf = FeasibleCounterfactualSoftmax(model.coef_, model.intercept_, X=X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub)
                    elif modeldesc == "dectree":
                        cf = FeasibleCounterfactualOfDecisionTree(model, X=X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub)
                    xcf = cf.compute_counterfactual(x_orig_orig, y_target=y_target, use_density_constraints=False)

                    if xcf is None:
                        results["notFound"] += 1
                        continue

                    # Compute counterfactual of perturbed sample
                    x_perturb = perturb(x_orig_orig)  # Perturb original data point
                    x_perturb_t = pca.transform([x_perturb]) if pca is not None else [x_perturb]
                    if model.predict(x_perturb_t) != y_orig:    
                        print("Perturbed sample is missclassified")

                    x_perturbed_cf = cf.compute_counterfactual(x_perturb, y_target=y_target, use_density_constraints=False)

                    if x_perturbed_cf is None:
                        results["notFound"] += 1
                        continue

                    # Compute a plausible counterfatual
                    cf2 = None
                    if modeldesc == "glvq":
                        cf2 = LvqCounterfactual(model, X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
                    elif modeldesc == "gmlvq":
                        cf2 = MatrixLvqCounterfactual(model, X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
                    elif modeldesc == "logreg":
                        cf2 = FeasibleCounterfactualSoftmax(model.coef_, model.intercept_, X=X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
                    elif modeldesc == "dectree":
                        cf2 = FeasibleCounterfactualOfDecisionTree(model, X=X_, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
                    xcf2 = cf2.compute_counterfactual(x_orig_orig, y_target=y_target, use_density_constraints=True)
                    if xcf2 is None:
                        results["notFound"] += 1
                        continue

                    # Compute plausible counterfactual of perturbed sample
                    x_perturbed_cf2 = cf2.compute_counterfactual(x_perturb, y_target=y_target, use_density_constraints=True)
                    if x_perturbed_cf2 is None:
                        results["notFound"] += 1
                        continue

                    results["found"] += 1

                    # Evaluate & store results
                    original_data.append(x_orig_orig)
                    original_data_labels.append(y_orig)
                    cfs_with_density_constraint.append(xcf2)
                    cfs_without_density_constraint.append(xcf)
                    cfs_perturbed_with_density_constraint.append(x_perturbed_cf2)
                    cfs_perturbed_without_density_constraint.append(x_perturbed_cf)
                    cfs_target_label.append(y_target)

                    distances_with_density_constraint.append(np.sum(np.abs(x_orig_orig - xcf2)))    # Store distance before projecting it again for density estimation!
                    distances_without_density_constraint.append(np.sum(np.abs(x_orig_orig - xcf)))
                    distances_perturbed_with_density_constraint.append(np.sum(np.abs(x_perturb - x_perturbed_cf2)))
                    distances_perturbed_without_density_constraint.append(np.sum(np.abs(x_perturb - x_perturbed_cf)))

                    cf_perturbation_dist = compare_cf(xcf, x_perturbed_cf)     # Distance between counterfactual of perturned and original sample
                    cf_feasible_perturbation_dist = compare_cf(xcf2, x_perturbed_cf2)
                    scores_cf_perturbation_dist.append(cf_perturbation_dist)
                    scores_cf_feasible_perturbation_dist.append(cf_feasible_perturbation_dist)

                    if pca is not None:
                        xcf = pca.transform([xcf])
                        xcf2 = pca.transform([xcf2])
                        x_perturbed_cf = pca.transform([x_perturbed_cf])
                        x_perturbed_cf2 = pca.transform([x_perturbed_cf2])

                    scores_without_density_constraint.append(kde.score_samples(xcf.reshape(1, -1)))
                    scores_with_density_constraint.append(kde.score_samples(xcf2.reshape(1, -1)))
                    scores_perturbed_without_density_constraint.append(kde.score_samples(xcf.reshape(1, -1)))
                    scores_perturbed_with_density_constraint.append(kde.score_samples(xcf2.reshape(1, -1)))

            if feature_mask is not None:
                dist_perturbed_with_density_constraint.append(np.median(scores_cf_feasible_perturbation_dist))
                dist_perturbed_without_density_constraint.append(np.median(scores_cf_perturbation_dist))

            print(f"Feature mask: {feature_mask}")
            print(f"Not found {results['notFound']}/{results['notFound'] + results['found']}")

            print("Without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_cf_perturbation_dist), np.mean(scores_cf_perturbation_dist), np.var(scores_cf_perturbation_dist)))
            print("With density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_cf_feasible_perturbation_dist), np.mean(scores_cf_feasible_perturbation_dist), np.var(scores_cf_feasible_perturbation_dist)))


            print("Unperturbed")
            print("Without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_without_density_constraint), np.mean(scores_without_density_constraint), np.var(scores_without_density_constraint)))
            print("With density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_with_density_constraint), np.mean(scores_with_density_constraint), np.var(scores_with_density_constraint)))

            print("Distances without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(distances_without_density_constraint), np.mean(distances_without_density_constraint), np.var(distances_without_density_constraint)))
            print("Distances with density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(distances_with_density_constraint), np.mean(distances_with_density_constraint), np.var(distances_with_density_constraint)))


            print("Perturbed")
            print("Without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_perturbed_without_density_constraint), np.mean(scores_perturbed_without_density_constraint), np.var(scores_perturbed_without_density_constraint)))
            print("With density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_perturbed_with_density_constraint), np.mean(scores_perturbed_with_density_constraint), np.var(scores_perturbed_with_density_constraint)))

            print("Distances without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(distances_perturbed_without_density_constraint), np.mean(distances_perturbed_without_density_constraint), np.var(distances_perturbed_without_density_constraint)))
            print("Distances with density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(distances_perturbed_with_density_constraint), np.mean(distances_perturbed_with_density_constraint), np.var(distances_perturbed_with_density_constraint)))

        # Plot
        plt.figure()
        x = masked_features[1:]
        y_with_density = dist_perturbed_with_density_constraint
        y_without_density = dist_perturbed_without_density_constraint
        plt.plot(x, y_without_density, label="Closest counterfactual")
        plt.plot(x, y_with_density, label="Plausible counterfactual")
        plt.xlabel("Number of masked features")
        plt.ylabel("Dist Cf of original vs. perturbed sample")
        plt.legend()
        plt.show()
        #plt.savefig(f"exp_results/{modeldesc}_{datasetdesc}_perturbation_dist.pdf", dpi=500, bbox_inches='tight', pad_inches = 0)
