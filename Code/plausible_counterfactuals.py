# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
from sklearn_lvq import GlvqModel, GmlvqModel
from tree import get_leafs_from_tree


class HighDensityEllipsoids:
    def __init__(self, X, X_densities, cluster_probs, means, covariances, density_threshold=None):
        self.X = X
        self.X_densities = X_densities
        self.density_threshold = density_threshold if density_threshold is not None else float("-inf")
        self.cluster_probs = cluster_probs
        self.means = means
        self.covariances = covariances
        self.t = 0.9
        self.epsilon = 0

    def compute_ellipsoids(self):        
        return self.build_solve_opt()
    
    def _solve(self, prob):
        prob.solve(solver=cp.MOSEK, verbose=False)

    def build_solve_opt(self):
        n_ellipsoids = self.cluster_probs.shape[1]
        n_samples = self.X.shape[0]
        
        # Variables
        r = cp.Variable(n_ellipsoids, pos=True)

        # Construct constraints
        constraints = []
        for i in range(n_ellipsoids):
            mu_i = self.means[i]
            cov_i = np.linalg.inv(self.covariances[i])

            for j in range(n_samples):
                if self.X_densities[j][i] <= self.density_threshold:  # At least as good as a requested NLL
                    x_j = self.X[j,:]
                    
                    a = (x_j - mu_i)
                    b = np.dot(a, np.dot(cov_i, a))
                    constraints.append(b <= r[i])

        # Build the final program
        f = cp.Minimize(cp.sum(r))
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return r.value


class Ellipsoids:
    def __init__(self, means, covariances, r, density_estimator):
        self.means = means
        self.covariances = [np.linalg.inv(cov) for cov in covariances]
        self.r = r
        self.n_ellipsoids = self.r.shape[0]
        self.density_estimator = density_estimator
    
    def score_samples(self, X):
        print(X.shape)
        pred = []

        pred = self.density_estimator.score_samples(X) >= -10

        return np.array(pred).astype(np.int)


class FeasibleCounterfactualSoftmax:
    def __init__(self, w, b, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_threshold=-85):
        self.w = w
        self.b = b

        self.kernel_var = 0.2#0.5   # Kernel density estimator
        self.X = X
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = np.eye(self.X.shape[1]) if projection_matrix is None else projection_matrix
        self.projection_mean_sub = np.zeros(self.X.shape[1]) if projection_mean_sub is None else projection_mean_sub
        self.density_constraint = False

        self.gmm_cluster_index = 0
        self.min_density = density_threshold
        self.epsilon = 1e-1

    def _build_constraints(self, var_x, y):
        constraints = []
        if self.w.shape[0] > 1:
            for i in range(self.w.shape[0]):
                if i != y:
                    constraints += [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ (self.w[i,:] - self.w[y,:]) + (self.b[i] - self.b[y]) + self.epsilon <= 0]
        else:
            if y == 0:
                return [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ self.w.reshape(-1, 1) + self.b + self.epsilon <= 0]
            else:
                return [(self.projection_matrix @ (var_x - self.projection_mean_sub)).T @ self.w.reshape(-1, 1) + self.b - self.epsilon >= 0]

        return constraints

    def compute_counterfactual(self, x, y_target, regularizer="l1", use_density_constraints=False): 
        self.density_constraint = use_density_constraints       
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        xcf = None
        s = float("inf")
        for i in range(self.gmm_weights.shape[0]):
            try:
                self.gmm_cluster_index = i
                xcf_ = self.build_solve_opt(x, y_target, mad)
                if xcf_ is None:
                    continue

                s_ = None
                if regularizer == "l1":
                    s_ = np.sum(np.abs(xcf_ - x))
                else:
                    s_ = np.linalg.norm(xcf_ - x, ord=2)

                if s_ <= s:
                    s = s_
                    xcf = xcf_
            except Exception as ex:
                print(ex)
        return xcf
    
    def _solve(self, prob):
        prob.solve(solver=cp.MOSEK, verbose=False)

    def build_solve_opt(self, x_orig, y, mad=None):
        dim = x_orig.shape[0]
        n_samples = self.X.shape[0]
        
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        t = cp.Variable()

        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)
        C = 1
        C_diag = C * np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(x, y)

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            w_i = self.gmm_weights[i]
            cov = self.gmm_covariances[i]
            cov = np.linalg.inv(cov)

            constraints += [cp.quad_form(self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov) - self.ellipsoids_r[i] + self.epsilon <= 0] # Numerically much more stable than the explicit density omponent constraint

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:           
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
        
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return x.value


class FeasibleCounterfactualOfDecisionTree:
    def __init__(self, model, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_threshold=-85):
        self.model = model

        self.kernel_var = 0.2  # Kernel density estimator
        self.X = X
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = np.eye(self.X.shape[1]) if projection_matrix is None else projection_matrix
        self.projection_mean_sub = np.zeros(self.X.shape[1]) if projection_mean_sub is None else projection_mean_sub
        self.density_constraint = False

        self.gmm_cluster_index = 0
        self.min_density = density_threshold
        self.epsilon = 0
        self.epsilon_plausibility = 1e-5

    def _solve(self, prob):
        prob.solve(solver=cp.MOSEK, verbose=False)

    def _build_constraints(self, var_x, y_target, path_to_leaf):
        constraints = []

        for j in range(0, len(path_to_leaf) - 1):
            feature_id = path_to_leaf[j][1]
            threshold = path_to_leaf[j][2]
            direction = path_to_leaf[j][3]

            if direction == "<":
                constraints.append((self.projection_matrix @ var_x)[feature_id] + self.epsilon <= threshold)
            elif direction == ">":
                constraints.append((self.projection_matrix @ var_x)[feature_id] - self.epsilon >= threshold)

        return constraints

    def build_solve_opt(self, x_orig, y, path_to_leaf, mad=None):
        dim = x_orig.shape[0]
        n_samples = self.X.shape[0]
        
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        t = cp.Variable()

        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)
        C = 1
        C_diag = C * np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(x, y, path_to_leaf)

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            cov = self.gmm_covariances[i]
            cov = np.linalg.inv(cov)

            constraints += [cp.quad_form(self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov) - self.ellipsoids_r[i] + self.epsilon_plausibility <= 0] # Numerically much more stable than the explicit density omponent constraint

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:  
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
        
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return x.value

    def compute_counterfactual(self, x, y_target, regularizer="l1", use_density_constraints=False):  
        self.density_constraint = use_density_constraints
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        xcf = None
        s = float("inf")
        
        # Find all paths that lead to a valid (but not necessarily feasible) counterfactual
        # Collect all leafs
        leafs = get_leafs_from_tree(self.model.tree_, classifier=True)

        # Filter leafs for predictions
        leafs = list(filter(lambda x: x[-1][2] == y_target, leafs))

        if len(leafs) == 0:
            raise ValueError("Tree does not has a path/leaf yielding the requested outcome specified in 'y_target'")
        
        # For each leaf: Compute feasible counterfactual
        for path_to_leaf in leafs:
            for i in range(self.gmm_weights.shape[0]):
                try:
                    self.gmm_cluster_index = i
                    xcf_ = self.build_solve_opt(x, y_target, path_to_leaf, mad)
                    if xcf_ is None:
                        continue

                    s_ = None
                    if regularizer == "l1":
                        s_ = np.sum(np.abs(xcf_ - x))
                    else:
                        s_ = np.linalg.norm(xcf_ - x, ord=2)

                    if s_ <= s:
                        s = s_
                        xcf = xcf_
                except Exception as ex:
                    print(ex)
        return xcf


class LvqCounterfactualBase(ABC):
    def __init__(self, model, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_constraint=True, density_threshold=-85):
        self.model = model
        self.prototypes = model.w_
        self.labels = model.c_w_

        self.kernel_var = 0.2#0.5   # Kernel density estimator
        self.X = X
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = np.eye(self.X.shape[1]) if projection_matrix is None else projection_matrix
        self.projection_mean_sub = np.zeros(self.X.shape[1]) if projection_mean_sub is None else projection_mean_sub
        self.density_constraint = density_constraint

        self.gmm_cluster_index = 0
        self.min_density = density_threshold
        self.epsilon = 1e-1

        super(LvqCounterfactualBase, self).__init__()
            
    @abstractmethod
    def compute_counterfactual(self, x_orig, y_target, features_whitelist=None, mad=None):
        raise NotImplementedError()


class MatrixLvqCounterfactual(LvqCounterfactualBase):
    def __init__(self, model, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_constraint=True, density_threshold=-85):
        if not isinstance(model, GmlvqModel):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GmlvqModel' but not of {type(model)}")

        super(MatrixLvqCounterfactual, self).__init__(model, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix, projection_mean_sub, density_constraint, density_threshold)
    
    def _solve(self, prob):
        prob.solve(solver=cp.MOSEK, verbose=False)

    def _build_omega(self):
        return np.dot(self.model.omega_.T, self.model.omega_)

    def _compute_counterfactual_target_prototype(self, x_orig, target_prototype, other_prototypes, y_target, features_whitelist=None, mad=None):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        
        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = []

        p_i = target_prototype

        Omega = self._build_omega()

        G = []
        b = np.zeros(len(other_prototypes))
        k = 0
        for k in range(len(other_prototypes)):
            p_j = other_prototypes[k]
            G.append(0.5 * np.dot(Omega, p_j - p_i))
            b[k] = -0.5 * (np.dot(p_i, np.dot(Omega, p_i)) - np.dot(p_j, np.dot(Omega, p_j))) - self.epsilon
        G = np.array(G)

        # Density constraints
        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            w_i = self.gmm_weights[i]
            cov = self.gmm_covariances[i]
            cov = np.linalg.inv(cov)

            constraints += [cp.quad_form(self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov) - self.ellipsoids_r[i] + self.epsilon <= 0] # Numerically much more stable than the explicit density omponent constraint

        # If requested, fix the values of some features/dimensions
        A = None
        a = None
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(self.dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            A = np.array(A)
            a = np.array(a)

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [G @ (self.projection_matrix @ (x - self.projection_mean_sub)) <= b, Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T @ x)  # Minimize L2 distance
            constraints += [G @ (self.projection_matrix @ (x - self.projection_mean_sub)) <= b]
        
        if A is not None and a is not None:
            constraints += [A @ x == a]
        
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)
        
        return x.value

    def compute_counterfactual(self, x_orig, y_target, features_whitelist=None, mad=None, use_density_constraints=False):
        xcf = None
        xcf_dist = float("inf")
        self.density_constraint = use_density_constraints
        mad = np.ones(x_orig.shape)

        dist = lambda x: np.linalg.norm(x - x_orig, 2)
        if mad is not None:
            dist = lambda x: np.dot(mad, np.abs(x - x_orig))
        
        # Search for suitable prototypes
        target_prototypes = []
        other_prototypes = []
        for p, l in zip(self.prototypes, self.labels):
            if l == y_target:
                target_prototypes.append(p)
            else:
                other_prototypes.append(p)
        
        # Compute a counterfactual for each prototype
        for i in range(len(target_prototypes)):
            try:
                if self.density_constraint is True:
                    for i in range(self.gmm_weights.shape[0]):
                        self.gmm_cluster_index = i

                        xcf_ = self._compute_counterfactual_target_prototype(x_orig, target_prototypes[i], other_prototypes, y_target, features_whitelist, mad)
                        xcf_proj = self.projection_matrix @ (xcf_ - self.projection_mean_sub)
                        ycf_ = self.model.predict([xcf_proj])[0]

                        if ycf_ == y_target:
                            if dist(xcf_) < xcf_dist:
                                xcf = xcf_
                                xcf_dist = dist(xcf_)
                else:
                    xcf_ = self._compute_counterfactual_target_prototype(x_orig, target_prototypes[i], other_prototypes, y_target, features_whitelist, mad)
                    xcf_proj = self.projection_matrix @ (xcf_ - self.projection_mean_sub)
                    ycf_ = self.model.predict([xcf_proj])[0]

                    if ycf_ == y_target:
                        if dist(xcf_) < xcf_dist:
                            xcf = xcf_
                            xcf_dist = dist(xcf_)
            except Exception as ex:
                print(ex)
        
        return xcf


class LvqCounterfactual(MatrixLvqCounterfactual):
    def __init__(self, model, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix=None, projection_mean_sub=None, density_constraint=True, density_threshold=-85):
        if not isinstance(model, GlvqModel):
            raise TypeError(f"model has to be an instance of 'sklearn_lvq.GlvqModel' but not of {type(model)}")
        
        self.dim = model.w_[0].shape[0]

        LvqCounterfactualBase.__init__(self, model, X, ellipsoids_r, gmm_weights, gmm_means, gmm_covariances, projection_matrix, projection_mean_sub, density_constraint, density_threshold) # Note: We can not call the constructor of the parent class because it expects a GmlvqModel
    
    def _build_omega(self):
        return np.eye(self.dim)
