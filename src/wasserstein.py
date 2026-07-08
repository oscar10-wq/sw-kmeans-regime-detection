
"""
Sliced Wasserstein k-means clustering for financial regime detection.

Implements two variants:
  (1) Original SW clustering — Luan et al. (2025), "Automated regime classification in
      multidimensional time series data using sliced Wasserstein k-means clustering."
  (2) SW clustering with UnifOrtho projections — Bardenet et al. (2025), "Repulsive
      Monte Carlo on the sphere for the sliced Wasserstein distance."
"""

import numpy as np 
import pandas as pd 
import math
import time
import sympy as sp
import random
import importlib
import src.metrics as mt
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats as stats


debug = False


# ---------------------------------------------------------------------------
# Core distribution classes
# ---------------------------------------------------------------------------

class EmpiricalDistribution:
    def __init__(self, l_m: np.ndarray):
        """
        l_m: np.array of shape (h1, d) where h1 is the number of points, d is dimension

        Parameters
        ----------
        l_m : np.ndarray of shape (h1, d)
            h1 support points in d dimensions.
        """
        self.l_m = np.asarray(l_m)
        self.h1, self.d = self.l_m.shape
        self.weights = np.ones(self.h1) / self.h1

    def project(self, theta: np.ndarray) -> np.ndarray:
        """Project support points onto the 1-D line defined by theta."""

        return np.sort(self.l_m @ theta)

class ProjectedDistribution:
    '''
    Create projections of high-dimensional distributions
    '''
    def __init__(self, x_points: np.ndarray):
        self.x_points = np.asarray(x_points)
        self.h1 = len(x_points)
        self.weights = np.ones(self.h1) / self.h1
        self.sorted_atoms = np.sort(self.x_points)
        self.mean_atoms = np.mean(self.x_points)
        self.var_atoms = np.var(self.x_points)

    def return_sorted_atoms(self) -> np.ndarray:
        return self.sorted_atoms

    def return_mean(self) -> float:
        return self.mean_atoms

    def return_variance(self) -> float:
        return self.var_atoms

# ---------------------------------------------------------------------------
# Sliced Wasserstein distance
# ---------------------------------------------------------------------------

class sliced_wasserstein_distance:
    """Vectorized sliced Wasserstein-p distance between two sets of projected distributions."""

    def __init__(self, projected_distributions_1, projected_distributions_2, p: int):
        self.dist1 = projected_distributions_1
        self.dist2 = projected_distributions_2
        self.p = p

    def compute_distance_matrix(self) -> float:
        # Extract the pre-sorted arrays into 2D matrices of shape (L, h1)
        X1 = np.array([d.return_sorted_atoms() for d in self.dist1])
        X2 = np.array([d.return_sorted_atoms() for d in self.dist2])
        
        # Vectorized calculation across all L projections simultaneously!
        if self.p == 1:
            return float(np.mean(np.mean(np.abs(X1 - X2), axis=1)))
        elif self.p == 2:
            # np.mean( ... , axis=1) computes the mean across the h1 atoms.
            # The outer np.mean() computes the average across the L projections.
            return float(np.mean(np.sqrt(np.mean((X1 - X2)**2, axis=1))))
        else:
            #p norm
            return float(np.mean(np.mean(np.abs(X1 - X2)**self.p, axis=1)**(1/self.p)))
        
    
# ---------------------------------------------------------------------------
# Barycenter computation
# ---------------------------------------------------------------------------

def sliced_wasserstein_compute_barycenter(projected_distributions: list, p: int) -> list:
    """    
     Compute the SW-p barycenter of a set of projected distributions.

    Parameters
    ----------
    projected_distributions : list[list[ProjectedDistribution]]
        Length-M_k list; each element is a length-L list of ProjectedDistribution objects.
    p : int
        Wasserstein order (1 → median, 2 → mean).

    Returns
    -------
    list[ProjectedDistribution]
        Length-L list of ProjectedDistribution objects representing the barycenter.
    """

    M_k = len(projected_distributions) 
    if M_k == 0:
        return []
    

    L = len(projected_distributions[0]) 

    # OPTIMIZATION 3: Extract all data into a single 3D NumPy array of shape (M_k, L, h1) tensor
    X = np.array([[d.return_sorted_atoms() for d in dist_list] for dist_list in projected_distributions])
    
    # Compute the mean/median across the M_k distributions (axis=0) in one vectorized shot
    if p == 1:
        centroid_points = np.median(X, axis=0) # Resulting shape is (L, h1)
    else:
        #case p ==2 
        centroid_points = np.mean(X, axis=0)
        
    # Wrap the L centroid projection arrays back into ProjectedDistribution objects
    return [ProjectedDistribution(centroid_points[l]) for l in range(L)]


# ---------------------------------------------------------------------------
# Lifting transformation
# ---------------------------------------------------------------------------

def lifting_transformation(r_S: np.ndarray, h1: int, h2: int) -> np.ndarray:

    """
    Embed a return series into a sequence of overlapping local windows.

    Parameters
    ----------
    r_S : np.ndarray of shape (N, d)
        Log-return series with N observations in d dimensions.
    h1 : int
        Window length.
    h2 : int
        Stride between successive windows.

    Returns
    -------
    np.ndarray of shape (M, h1, d)
        Lifted sample array where M = floor((N − h1) / h2) + 1.
    """
     
    windows = sliding_window_view(r_S, window_shape=h1, axis=0)
    
    windows = windows.transpose(0, 2, 1)
    
    lifted_samples = windows[::h2]
    
    return lifted_samples.copy()


# ---------------------------------------------------------------------------
# Projection vector generation (UnifOrtho)
# ---------------------------------------------------------------------------

def unifortho_projection_vectors(S: np.ndarray, K: int, L: int, h1: int, h2: int) -> list:
    """
    Compute projected empirical distributions using UnifOrtho random projections.
    Generates L projection directions via repeated QR decomposition of standard
    normal matrices, then projects all lifted windows in a single tensor operation.  

    Parameters
    ----------
    S : np.ndarray of shape (N, d)
        Price (or level) series.
    K : int
        Number of clusters (unused here; kept for API consistency).
    L : int
        Number of projection directions.
    h1 : int
        Lifting window length.
    h2 : int
        Lifting stride.
    Returns
    -------
    list[list[ProjectedDistribution]]
        Shape (M, L): projected distributions for each lifted window.
    """ 
     

    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    # perform lifting transfo on returns 
    l_r_S = lifting_transformation(r_S, h1, h2)
    
    M = math.floor((N-(h1-h2))/h2)

    emp_dist = [] 
    for m in range(M):
        emp_dist.append(EmpiricalDistribution(l_r_S[m, :, :])) # l_r_S[m,:,:].shape == (h1, d)

    # Step 1: Generate L random orthogonal projection vectors using Monte Carlo (UnifOrtho) method
    d = r_S.shape[1]
    k = math.ceil(L/d)
    theta = []
    for i in range(k):
        Z = np.random.normal(size=(d, d))
        Q, R = np.linalg.qr(Z)
        lambda_i = np.diag(np.sign(np.diag(R)))
        U_i = Q @ lambda_i
        theta.extend(U_i.T)
    
    theta = np.array(theta)
    theta = theta.T
    theta = theta[:, :L]  #theta[:,:L].shape ==  (d, L)

    # Step 2: Projection and k-mean iteration : HAS TO BE DONE ONLY ONCE ==> CACHE 
    projected_emp_dist = [[ProjectedDistribution(emp_dist[m].project(theta[:, l])) for l in range(L)] for m in range(M)]
    return projected_emp_dist    


# ---------------------------------------------------------------------------
# Projection vector generation (UnifOrtho) OPTIMIZED 
# ---------------------------------------------------------------------------


def unifortho_projection_vectors_opt(S: np.ndarray, K: int, L: int, h1: int, h2: int)-> list:
    """
    Compute projected empirical distributions using UnifOrtho random projections.
    Generates L projection directions via repeated QR decomposition of standard
    normal matrices, then projects all lifted windows in a single tensor operation.  

    Parameters
    ----------
    S : np.ndarray of shape (N, d)
        Price (or level) series.
    K : int
        Number of clusters (unused here; kept for API consistency).
    L : int
        Number of projection directions.
    h1 : int
        Lifting window length.
    h2 : int
        Lifting stride.
    Returns
    -------
    list[list[ProjectedDistribution]]
        Shape (M, L): projected distributions for each lifted window.
    """ 


    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    l_r_S = lifting_transformation(r_S, h1, h2) # Shape: (M, h1, d)
    M = math.floor((N-(h1-h2))/h2)

    # Step 1: Generate L random orthogonal projection vectors (UnifOrtho)
    d = r_S.shape[1]
    k = math.ceil(L/d)
    theta = []
    for i in range(k):
        Z = np.random.normal(size=(d, d))
        Q, R = np.linalg.qr(Z) # O(d^3)
        lambda_i = np.diag(np.sign(np.diag(R)))
        U_i = Q @ lambda_i
        theta.extend(U_i.T)
    
    theta = np.array(theta).T[:, :L]  # Shape: (d, L)

    # =========================================================================
    # OPTIMIZATION 1: Vectorized Tensor Projection
    # =========================================================================
    # 1. Multiply all lifted samples by the projection matrix at once
    #    (M, h1, d) @ (d, L) --> (M, h1, L)
    projections = l_r_S @ theta
    
    # 2. Sliced Wasserstein requires sorted atoms. We sort along the h1 axis. (M, h1, L)
    sorted_projections = np.sort(projections, axis=1) 
    
    # 3. Transpose to (M, L, h1) so it matches our list comprehension iteration
    sorted_projections = sorted_projections.transpose(0, 2, 1)

    # 4. Instantiate objects instantly using the pre-calculated, pre-sorted arrays
    projected_emp_dist = [[ProjectedDistribution(sorted_projections[m, l]) for l in range(L)] for m in range(M)]
    
    return projected_emp_dist



# ---------------------------------------------------------------------------
# K-means clustering loop
# ---------------------------------------------------------------------------


def sliced_wasserstein_clustering_conv_loop(projected_emp_dist: list, K: int, M: int, L: int, epsilon: float) -> tuple:
    """
    Sliced Wasserstein k-means with vectorized assignment and barycenter updates.
    Parameters
    ----------
    projected_emp_dist : list[list[ProjectedDistribution]]
        Pre-computed projected distributions, shape (M, L).
    K : int
        Number of clusters.
    M : int
        Number of lifted windows.
    L : int
        Number of projection directions.
    epsilon : float
        Convergence threshold on total centroid movement.

    Returns
    -------
    projected_emp_dist : list[list[ProjectedDistribution]]
    centroids : list[list[ProjectedDistribution]]
        Final centroid distributions, shape (K, L).
    labels : np.ndarray of shape (M,)
        Cluster assignment for each lifted window.
    
    """ 
    # Initialize K random centroids (1D distributions) for k-means clustering. Choose one distrbituion per cluster for initialization.
    centroids = [0]*K
    for k in range(K):   
        centroids[k] = projected_emp_dist[random.randint(0, M-1)]
    labels = np.full(M, -1) 
    old_centroids = None

    max_iterations = 50 # Set a proper max limit
    for iteration in range(max_iterations):
        #print(f"--- Iteration {iteration + 1} ---")
        
        # Keep track of old labels to check for convergence

        if old_centroids is not None:
            # Check for convergence: if centroids haven't changed significantly, we can stop
            centroid_changes = sum([sliced_wasserstein_distance(old_centroids[k], centroids[k], p=2).compute_distance_matrix() for k in range(K)])
            
            if centroid_changes < epsilon:
                #print("Convergence reached based on centroid changes.")
                break
        
        old_labels = labels.copy()
        old_centroids = centroids.copy() 

        # ==========================================
        # STEP 1: EXPECTATION (Assign to closest centroid)
        # ==========================================
        for m in range(M):
            # Calculate distance to all K centroids
            # Note: k-means theoretically uses squared Wasserstein-2 distance (p=2) 
            # to match the arithmetic mean used in the barycenter step!
            distances_to_centroids = [
                sliced_wasserstein_distance(projected_emp_dist[m], centroids[k], p=2).compute_distance_matrix() 
                for k in range(K)
            ]
            # Assign to the closest centroid
            labels[m] = np.argmin(distances_to_centroids)
            
        # STEP 2: MAXIMIZATION (Update Centroids)
        # ==========================================
        for k in range(K):
            # Gather all projected distributions currently assigned to cluster k
            cluster_k_distributions = [projected_emp_dist[m] for m in range(M) if labels[m] == k]
            
            if len(cluster_k_distributions) > 0:
                centroids[k] = sliced_wasserstein_compute_barycenter(cluster_k_distributions, p=2)

            else:
                #print(f"Warning: Cluster {k} became empty. Reinitializing.")
                # Standard k-means fallback: pick a new random point if a cluster dies
                centroids[k] = projected_emp_dist[random.randint(0, M-1)]

    #print("Finished clustering after", iteration + 1, "iterations.", "Final centroid changes:", centroid_changes)
    return projected_emp_dist, centroids, labels


def sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist: list, K: int, M: int, L: int, epsilon: float) -> tuple:
    
    """
    Sliced Wasserstein k-means with vectorized assignment and barycenter updates.
    Parameters
    ----------
    projected_emp_dist : list[list[ProjectedDistribution]]
        Pre-computed projected distributions, shape (M, L).
    K : int
        Number of clusters.
    M : int
        Number of lifted windows.
    L : int
        Number of projection directions.
    epsilon : float
        Convergence threshold on total centroid movement.

    Returns
    -------
    projected_emp_dist : list[list[ProjectedDistribution]]
    centroids : list[list[ProjectedDistribution]]
        Final centroid distributions, shape (K, L).
    labels : np.ndarray of shape (M,)
        Cluster assignment for each lifted window.
    
    """ 
     
    # Initialize K random centroids
    centroids = [projected_emp_dist[random.randint(0, M - 1)] for _ in range(K)]    
    labels = np.full(M, -1) 
    old_centroids = None

    # =========================================================================
    # OPTIMIZATION 2: Pre-extract data into a 3D NumPy array for blazing fast K-means
    # =========================================================================
    # Extract shape: (M, L, h1)
    h1 = len(projected_emp_dist[0][0].return_sorted_atoms())
    X = np.empty((M, L, h1))
    for m in range(M):
        for l in range(L):
            X[m, l, :] = projected_emp_dist[m][l].return_sorted_atoms()

    max_iterations = 50 
    for iteration in range(max_iterations):
        
        # =========================================================================
        # OPTIMIZATION 3: Extract current centroids into array shape (K, L, h1)
        # =========================================================================
        C = np.empty((K, L, h1))
        for k in range(K):
            for l in range(L):
                C[k, l, :] = centroids[k][l].return_sorted_atoms()

        if old_centroids is not None:
            # Vectorized Convergence Check
            # Mean Squared Error between old and new centroids
            C_old = np.empty((K, L, h1))
            for k in range(K):
                for l in range(L):
                    C_old[k, l, :] = old_centroids[k][l].return_sorted_atoms()
            
            centroid_changes = np.sum(np.mean(np.sqrt(np.mean((C_old - C)**2, axis=2)), axis=1))
            if centroid_changes < epsilon:
                break

        old_labels = labels.copy()
        old_centroids = centroids.copy() 

        # =========================================================================
        # OPTIMIZATION 4: EXPECTATION (Assign to closest centroid)
        # =========================================================================
        
        # --- NEW VECTORIZED EXPECTATION ---
        # Calculate distances using Broadcasting: X is (M, 1, L, h1), C is (1, K, L, h1)
        # This single line computes SW-2 distance for ALL points against ALL centroids!
        diff_sq = (X[:, None, :, :] - C[None, :, :, :]) ** 2
        
        # SW_2 distance: mean over h1 (axis 3), sqrt, mean over L (axis 2)
        sw_dist = np.mean(np.sqrt(np.mean(diff_sq, axis=3)), axis=2) # Shape: (M, K)
        
        # Assign labels instantaneously
        labels = np.argmin(sw_dist, axis=1)
            
        # =========================================================================
        # OPTIMIZATION 5: MAXIMIZATION (Update Centroids)
        # =========================================================================
        # --- NEW VECTORIZED MAXIMIZATION ---
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                # Barycenter for p=2 is simply the mathematical mean across assigned distributions
                # X[mask] has shape (M_k, L, h1). Mean along axis 0 gives (L, h1)
                barycenter_matrix = np.mean(X[mask], axis=0)

                # Repackage into ProjectedDistribution objects for the next iteration
                centroids[k] = [ProjectedDistribution(barycenter_matrix[l]) for l in range(L)]

            else:
                #print(f"Warning: Cluster {k} became empty. Reinitializing.")
                centroids[k] = projected_emp_dist[random.randint(0, M-1)]

    return projected_emp_dist, centroids, labels



# ---------------------------------------------------------------------------
# Original Sliced Wasserstein function from Luan et al. (2025) 
# ---------------------------------------------------------------------------

def sliced_wasserstein_clustering(r_S, K, projection_vectors, epsilon, h1, h2):
    #TODO Implement the original Sliced Wasserstein clustering function as described in Luan et al. (2025) "Automated regime classification in multidimensional time series data using sliced Wasserstein k-mean clustering"
    return 0



# ---------------------------------------------------------------------------
# Sliced Wasserstein clustering algorithm with UnifOrtho projections
# ---------------------------------------------------------------------------

def sliced_wasserstein_clustering_unifortho(S: np.ndarray, K: int, L: int, epsilon: float, h1: int, h2: int) -> tuple:
    """
    Full SW k-means pipeline using UnifOrtho projections.

    Parameters
    ----------
    S : np.ndarray of shape (N, d)
        Price series.
    K : int
        Number of regimes.
    L : int
        Number of projection directions.
    epsilon : float
        Convergence tolerance.
    h1 : int
        Lifting window length.
    h2 : int
        Lifting stride.

    Returns
    -------
    (projected_emp_dist, centroids, labels)
    """

    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    M = math.floor((N-(h1-h2))/h2)
    
    # Step 1 and 2: Projection and k-mean iteration (cache the projected distributions for all lifted samples)
    projected_emp_dist = unifortho_projection_vectors_opt(S, K, L, h1, h2)

    # Step 3: Sliced Wasserstein k-means clustering with convergence loop
    return sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon)



def max_acc_unifortho_sim(N_S: int, S: np.ndarray, r_s_true_regime, K: int, L: int, epsilon: float, h1: int, h2: int, test: bool = False) -> tuple:
    """
    Run N_S independent k-means initialisations and return the run with highest accuracy.

    Parameters
    ----------
    N_S : int
        Number of random restarts.
    S : np.ndarray of shape (N, d)
        Price series.
    r_s_true_regime : array-like
        Ground-truth regime labels aligned with the lifted windows.
    K, L, epsilon, h1, h2
        Passed through to the clustering routine.
    test : bool
        If True, print distributional statistics for each discovered regime.
    """
    
    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    # perform lifting transfo on returns
    M = math.floor((N-(h1-h2))/h2)
    #start = time.perf_counter()
    projected_emp_dist = unifortho_projection_vectors_opt(S, K, L, h1, h2)
    #end = time.perf_counter()
    #print("Non_Optimized_Projection_Vectors Time", end- start)
    best_accuracy = 0
    #best_projected_emp_dist = None
    best_centroids = None
    best_labels = None
    for sim in range(N_S):
        _, centroids, labels = sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon)
        print(f"Simulation {sim + 1}/{N_S} completed. Evaluating accuracy...")
        accuracy = mt.total_accuracy(S, r_s_true_regime, labels, h1, h2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_centroids = centroids
            best_labels = labels
    
    ## TESTING FOR METRICS TO DIFFERENTIAL BULL/BEAR REGIMES 
    if (test): 
        for k in range(K):
            # Assuming your 'projected' object has a way to get the raw array of returns/atoms
            # e.g., projected.get_atoms() or projected.samples
            
            means = []
            variances = []
            skews = []
            cvars = []
            
            for projected in best_centroids[k]:
                sorted_atoms = projected.return_sorted_atoms() 

                means.append(np.mean(sorted_atoms))
                variances.append(np.var(sorted_atoms))
                skews.append(stats.skew(sorted_atoms))
                
                # 5% Expected Shortfall (Conditional VaR)
                cvars.append(np.mean(sorted_atoms[:max(1, int(len(sorted_atoms)*0.05))])) 

            print(f'--- Regime {k} ---')
            print(f'Mean: {np.mean(means)}')
            print(f'Variance: {np.mean(variances)}')
            print(f'Skewness: {np.mean(skews)}')
            print(f'CVaR (5%): {np.mean(cvars)}')
            print('-'*20)

    return projected_emp_dist, best_centroids, best_labels

def max_mccd_unifortho_sim(N_S: int, S: np.ndarray, K: int, L: int, epsilon: float, h1: int, h2: int, metric: str = "CVaR") -> tuple:
    """
    Run N_S independent k-means initialisations and return the run with highest
    mean centroid-to-centroid distance (MCCD), then relabel regimes semantically.

    Parameters
    ----------
    N_S : int
        Number of random restarts.
    S : np.ndarray of shape (N, d)
        Price series.
    K, L, epsilon, h1, h2
        Passed through to the clustering routine.
    metric : {"CVaR", "MeanVar"}
        Criterion used to assign bull/bear labels (see `choose_label`).
    """

    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    # perform lifting transfo on returns
    M = math.floor((N-(h1-h2))/h2)
    #start = time.perf_counter()
    projected_emp_dist = unifortho_projection_vectors_opt(S, K, L, h1, h2)
    #end = time.perf_counter()
    #print("Non_Optimized_Projection_Vectors Time", end- start)
    best_mccd = 0
    #best_projected_emp_dist = None
    best_centroids = None
    best_labels = None
    for sim in range(N_S):
        _, centroids, labels = sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon)
        if debug:
            print(f"Simulation {sim + 1}/{N_S} completed. Evaluating accuracy...")
        mccd = mt.mean_centroid_centroid_distance(centroids, K, p=2)
        if mccd > best_mccd:
            best_mccd = mccd
            best_centroids = centroids
            best_labels = labels
    
    #associate label 0 to bearish associate label 1 to bullish
    #new_best_centroids = best_centroids 
    #new_best_labels = best_labels
    new_best_centroids, new_best_labels = choose_label(best_centroids, best_labels, metric, K)
    return projected_emp_dist, new_best_centroids, new_best_labels


# ---------------------------------------------------------------------------
# Regime labelling
# ---------------------------------------------------------------------------
import numpy as np
from scipy import stats

def choose_label(best_centroids: list, best_labels: np.ndarray, metric: str, K: int) -> tuple:
    """
    Canonicalise cluster labels so that:
    - K=2: label 0 is Bearish, label 1 is Bullish.
    - K=3: label 0 is Bearish, label 1 is Neutral/Transition, label 2 is Bullish.

    Parameters
    ----------
    best_centroids : list[list[ProjectedDistribution]]
    best_labels : np.ndarray of shape (M,)
    metric : {"CVaR", "MeanVar", "Skewness"}
    K : int
        Number of clusters (Supports 2 and 3).
        
    Returns
    -------
    (best_centroids, best_labels) with labels canonicalised.
    """

    # 1. Compute metrics for each cluster
    dic_metrics = {}
    
    for k in range(K):
        means = []
        variances = []
        skews = []
        cvars = []

        for projected in best_centroids[k]:
            sorted_atoms = projected.return_sorted_atoms() 

            means.append(np.mean(sorted_atoms))
            variances.append(np.var(sorted_atoms))
            skews.append(stats.skew(sorted_atoms))
            
            # 5% Expected Shortfall (Conditional VaR)
            cvars.append(np.mean(sorted_atoms[:max(1, int(len(sorted_atoms)*0.05))])) 

        dic_metrics[k] = {
            'mean': np.mean(means),
            'var': np.mean(variances),
            'skew': np.mean(skews),
            'cvar': np.mean(cvars)
        }

    # 2. Determine the sorted order of cluster indices based on the metric
    if metric == "CVaR":
        # More negative/lower CVaR = Worse (Bearish). 
        # Sorting ascending means: [Worst CVaR, Middle CVaR, Best CVaR]
        sorted_clusters = sorted(range(K), key=lambda k: dic_metrics[k]['cvar'])

    elif metric == "MeanVar":
        # For K=3, we look for a risk-adjusted return proxy. 
        # Bearish: Low mean, High variance. Bullish: High mean, Low variance.
        # We can sort primarily by Mean (ascending). If means are identical, higher variance comes first.
        sorted_clusters = sorted(range(K), key=lambda k: (dic_metrics[k]['mean'], -dic_metrics[k]['var']))
        
    elif metric == "Skewness":
        # Alternative metric: Bearish regimes often have strong negative skewness (market crashes).
        sorted_clusters = sorted(range(K), key=lambda k: dic_metrics[k]['skew'])
        
    else:
        print(f"Metric '{metric}' not recognized. Keeping original labels.")
        return best_centroids, best_labels

    # 3. Permute the centroids and labels based on the new ranking
    # sorted_clusters[0] becomes label 0 (Bear)
    # sorted_clusters[1] becomes label 1 (Neutral if K=3, Bull if K=2)
    # sorted_clusters[2] becomes label 2 (Bull if K=3)
    
    # Reorder centroids
    new_centroids = [best_centroids[old_idx] for old_idx in sorted_clusters]
    
    # Map the old labels to the new rank labels using a mapping array
    # mapping[old_label] = new_label
    mapping = np.zeros(K, dtype=int)
    for new_label, old_label in enumerate(sorted_clusters):
        mapping[old_label] = new_label
        
    new_labels = mapping[best_labels]

    #print(f"Canonicalised K={K} clusters using {metric}. Order mapping (old -> new): {sorted_clusters}")
    return new_centroids, new_labels


# ---------------------------------------------------------------------------
# Implied regime probabilities
# ---------------------------------------------------------------------------

def compute_implied_proba(projected_emp_dist: list, centroids: list, labels: np.ndarray, tau=None, tau_gradient=None, lookback: int = 5, use_gradient: bool = False, gradient_weight: float = 0.3) -> tuple:
    
    """
    Compute soft regime probabilities and a regime-switch signal for the latest window.
    
    Probabilities are derived from a softmax over SW-2 distances to each centroid,
    calibrated by temperature tau. A Bayesian update incorporates the empirical
    transition matrix, and an optional gradient term captures directional momentum
    toward an alternative regime.

    Parameters
    ----------
    projected_emp_dist : list[list[ProjectedDistribution]]
        Shape (M, L).
    centroids : list[list[ProjectedDistribution]]
        Shape (K, L).
    labels : np.ndarray of shape (M,)
        Hard cluster assignments from the k-means step.
    tau : float, optional
        Softmax temperature. Defaults to half the mean inter-centroid distance.
    tau_gradient : float, optional
        Temperature for the gradient signal softmax.
    lookback : int
        Number of recent windows used for the gradient slope estimate.
    use_gradient : bool
        If True, blend the Bayesian posterior with a trajectory-based signal.
    gradient_weight : float
        Mixing weight in [0, 1] for the gradient signal.

    Returns
    -------
    proba_matrix : np.ndarray of shape (M, K)
        Per-window regime probabilities.
    switch_proba : float
        Probability that the latest window is transitioning away from its current regime.
    transition_matrix : np.ndarray of shape (K, K)
        Empirical regime transition matrix estimated from label history.
    posterior : np.ndarray of shape (K,)
        Bayesian posterior for the latest window.
    
    """
    M = len(projected_emp_dist)
    K = len(centroids)

    # --- Step 1: Compute SW distance from every sample to every centroid ---
    # Reuse your optimized structure: extract into arrays
    h1_len = len(projected_emp_dist[0][0].return_sorted_atoms())
    L = len(projected_emp_dist[0])

    X = np.empty((M, L, h1_len))
    for m in range(M):
        for l in range(L):
            X[m, l, :] = projected_emp_dist[m][l].return_sorted_atoms()

    C = np.empty((K, L, h1_len))
    for k in range(K):
        for l in range(L):
            C[k, l, :] = centroids[k][l].return_sorted_atoms()

    # Vectorized SW-2 distance: (M, K)
    diff_sq = (X[:, None, :, :] - C[None, :, :, :]) ** 2
    dist_matrix = np.mean(np.sqrt(np.mean(diff_sq, axis=3)), axis=2)


    # --- Step 2: Calibrate temperature if not provided ---
    if tau is None:
        # Use half the mean inter-centroid distance as default
        centroid_dists = []
        for i in range(K):
            for j in range(i + 1, K):
                d_ij = np.mean(np.sqrt(np.mean((C[i] - C[j])**2, axis=1)))
                centroid_dists.append(d_ij)
        tau = 0.5 * np.mean(centroid_dists) if centroid_dists else 1.0

    # --- Step 3: Softmax over negative distances ---
    neg_scaled = -dist_matrix / tau
    neg_scaled -= neg_scaled.max(axis=1, keepdims=True)  # numerical stability
    exp_vals = np.exp(neg_scaled)
    proba_matrix = exp_vals / exp_vals.sum(axis=1, keepdims=True)  # (M, K)

    # --- Step 4: Empirical transition matrix from label history ---
    transition_matrix = np.zeros((K, K))
    for m in range(M - 1):
        transition_matrix[labels[m], labels[m + 1]] += 1
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    transition_matrix /= row_sums

    
    # --- Step 5: Bayesian combination for the latest sample ---
    current_regime = labels[-1]
    # Prior: transition probabilities from current regime
    prior = transition_matrix[current_regime]  # (K,)
    # Likelihood: distance-based softmax for the last sample
    likelihood = proba_matrix[-1]  # (K,)
    # Posterior (multiply and renormalize)
    posterior = prior * likelihood
    posterior /= (posterior.sum() + 1e-12)  # avoid division by zero


    if use_gradient and M >= lookback:
        # Slope of distance to current centroid over recent samples
        # Positive slope = drifting away from current regime = weakening
        recent_dist_current = dist_matrix[-lookback:, current_regime]
        slope_current = np.polyfit(np.arange(lookback), recent_dist_current, deg=1)[0]

        # Slope of distance to each alternative centroid
        # Negative slope = approaching that regime = strengthening
        gradient_signal = np.zeros(K)
        for k in range(K):
            recent_dist_k = dist_matrix[-lookback:, k]
            slope_k = np.polyfit(np.arange(lookback), recent_dist_k, deg=1)[0]
            # Flip sign: negative slope (approaching) -> high score
            gradient_signal[k] = -slope_k

        # Normalize into a probability-like vector via softmax
        gradient_signal -= gradient_signal.max()
        if tau_gradient is None:
            tau_gradient = 0.5 * np.std(gradient_signal) if np.std(gradient_signal) > 0 else 1.0 #this for now but to be changed

    
        gradient_proba = np.exp(gradient_signal / tau_gradient)
        gradient_proba /= (gradient_proba.sum() + 1e-12)  # avoid division by zero

        # Blend: posterior = (1 - w) * bayesian_posterior + w * gradient_signal
        posterior = (1 - gradient_weight) * posterior + gradient_weight * gradient_proba
        posterior /= (posterior.sum() + 1e-12)  # avoid division by zero

    switch_proba = 1.0 - posterior[current_regime]

    return proba_matrix, switch_proba, transition_matrix, posterior

    

if __name__ == "__main__":
    # Example usage
    r_S = np.random.rand(100, 3) # 100 samples in 3 dimensions
    K = 3
    L = 6
    epsilon = 1e-6
    h1 = 10
    h2 = 9
   # _,_,_ = sliced_wasserstein_clustering_unifortho(r_S, K, L, epsilon, h1, h2)
    



 





