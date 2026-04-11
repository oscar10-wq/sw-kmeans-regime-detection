'''
Python file containing functions to generate synthetic data of 5 different types :
 - (1) Standard Brownian motion with drift and correlation parameter rho
 - (2) Brownian motion with two regimes (Bull and Bear) where the regimes differ in their drift, volatility but same correlation structure (rho_bull = rho_bear)
 - (3) Brownian motion with two regimes (Bull and Bear) where the regimes differ in their drift, volatility and correlation structure (rho_bull != rho_bear) 
 - (4) Brownian motion with three regimes (Bull, Bear 1 and Bear 2) where the regimes differ in their drift, volatility and correlation structure (rho_bull = rho_bear_1 != rho_bear_2)
 - (5) Brownian motion with three regimes (Bull, Bear and Moon) where the regimes differ in their drift, volatility and correlation structure (rho_bull != rho_bear != rho_moon) and the Moon regime has a make moons component to make it more challenging to classify.
'''



import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
from mpl_toolkits.mplot3d import Axes3D 


# (1) Function with one parameter theta
def brownian_motion(S0, theta, T, dt, d, rho):
    N = int(T/dt)
    delta = np.eye(d)
    mu = theta[0]
    sigma = theta[1]
    cov_matrix = sigma**2 * (delta + (1 - delta) * rho)
    t = np.linspace(0, T, N)
    r = np.random.multivariate_normal((mu - sigma**2/2)*dt*np.ones(d), cov_matrix*dt, N)
    S = np.zeros((N, d))
    S[0] = S0
    for i in range(1, N):
        S[i] = S[i-1] * np.exp(r[i])
    return t, S

# (2) Type A Synthetic Data generation function with two parameter theta_bull, theta_bear #nb clusters K = 2
def brownian_motion_mult_regime_A(S0, theta_bull, theta_bear, T, dt, d, rho, n_bear_periods=10, bear_duration_years=1.0):
    N = int(T/dt)
    delta = np.eye(d)
    mu_bull = theta_bull[0]
    sigma_bull = theta_bull[1]
    mu_bear = theta_bear[0]
    sigma_bear = theta_bear[1]
    
    cov_matrix_bull = sigma_bull**2 * (delta + (1 - delta) * rho)
    cov_matrix_bear = sigma_bear**2 * (delta + (1 - delta) * rho)
    
    regimes = np.zeros(N, dtype=int)

    #half year period for bear markets is 126*7 observations (7 observations a day and therefore 252*7/2 observations half a year)
    bear_len_steps = int(bear_duration_years *126*7) 
    
    # 2. Randomly assign Bear periods (Non-overlapping)
    bear_starts = []
    max_attempts = 5000 # Prevent infinite loops if too crowded
    attempts = 0

    while len(bear_starts) < n_bear_periods and attempts < max_attempts:
        # Pick a random start index
        start_idx = np.random.randint(0, N - bear_len_steps)
        
        # Check if it overlaps with any existing bear market interval
        overlap = False
        for s in bear_starts:
            if abs(start_idx - s) < bear_len_steps:
                overlap = True
                break
                
        # If no overlap, record it and update the regime mask
        if not overlap:
            bear_starts.append(start_idx)
            regimes[start_idx : start_idx + bear_len_steps] = 1
            
        attempts += 1
        
    if attempts == max_attempts:
        print("Warning: Could not fit all bear periods without overlap. Try reducing duration or number of periods.")

    # 3. Generate Returns (Vectorized for massive speed improvement!)
    drift_bull = (mu_bull - sigma_bull**2 / 2) * dt
    drift_bear = (mu_bear - sigma_bear**2 / 2) * dt
    
    # Generate full arrays for both scenarios at once
    r_bull = np.random.multivariate_normal(drift_bull * np.ones(d), cov_matrix_bull * dt, size=N)
    r_bear = np.random.multivariate_normal(drift_bear * np.ones(d), cov_matrix_bear * dt, size=N)
    
    # Select the correct return based on the regimes array
    r = np.where(regimes[:, None] == 0, r_bull, r_bear)
    
    # 4. Construct Prices (Vectorized cumulative sum)
    # S_t = S_0 * exp(sum(r))
    cumulative_returns = np.cumsum(r, axis=0)
    S = S0 * np.exp(cumulative_returns)
    
    t = np.linspace(0, T, N)
    
    # Returning regimes array is helpful for evaluating your k-means accuracy later!
    return t, S, bear_starts, regimes

# (3) Type B Synthetic Data generation function with two parameter theta_bull, theta_bear #nb clusters K = 2
def brownian_motion_mult_regime_B(S0, theta_bull, theta_bear, T, dt, d, rho_bull, rho_bear, n_bear_periods=10, bear_duration_years=1.0):
    N = int(T/dt)
    delta = np.eye(d)
    mu_bull = theta_bull[0]
    sigma_bull = theta_bull[1]
    mu_bear = theta_bear[0]
    sigma_bear = theta_bear[1]
    
    cov_matrix_bull = sigma_bull**2 * (delta + (1 - delta) * rho_bull)
    cov_matrix_bear = sigma_bear**2 * (delta + (1 - delta) * rho_bear)
    
    regimes = np.zeros(N, dtype=int)

    #half year period for bear markets is 126*7 observations (7 observations a day and therefore 252*7/2 observations half a year)
    bear_len_steps = int(bear_duration_years *126*7) 
    
    
    # 2. Randomly assign Bear periods (Non-overlapping)
    bear_starts = []
    max_attempts = 5000 # Prevent infinite loops if too crowded
    attempts = 0

    while len(bear_starts) < n_bear_periods and attempts < max_attempts:
        # Pick a random start index
        start_idx = np.random.randint(0, N - bear_len_steps)
        
        # Check if it overlaps with any existing bear market interval
        overlap = False
        for s in bear_starts:
            if abs(start_idx - s) < bear_len_steps:
                overlap = True
                break
                
        # If no overlap, record it and update the regime mask
        if not overlap:
            bear_starts.append(start_idx)
            regimes[start_idx : start_idx + bear_len_steps] = 1
            
        attempts += 1
        
    if attempts == max_attempts:
        print("Warning: Could not fit all bear periods without overlap. Try reducing duration or number of periods.")

    # 3. Generate Returns (Vectorized for massive speed improvement!)
    drift_bull = (mu_bull - sigma_bull**2 / 2) * dt
    drift_bear = (mu_bear - sigma_bear**2 / 2) * dt
    
    # Generate full arrays for both scenarios at once
    r_bull = np.random.multivariate_normal(drift_bull * np.ones(d), cov_matrix_bull * dt, size=N)
    r_bear = np.random.multivariate_normal(drift_bear * np.ones(d), cov_matrix_bear * dt, size=N)
    
    # Select the correct return based on the regimes array
    r = np.where(regimes[:, None] == 0, r_bull, r_bear)
    
    # 4. Construct Prices (Vectorized cumulative sum)
    # S_t = S_0 * exp(sum(r))
    cumulative_returns = np.cumsum(r, axis=0)
    S = S0 * np.exp(cumulative_returns)
    
    t = np.linspace(0, T, N)
    
    # Returning regimes array is helpful for evaluating your k-means accuracy later!
    return t, S, bear_starts, regimes


# (4) Type C Synthetic Data generation function with two parameter theta_bull, theta_bear #nb clusters K = 3
# Randomly alternating between regime bear with rho_bear = 1/2 and regime bear with rho_bear = -1/2 among the 10 half year bear periods. 
def brownian_motion_mult_regime_C(S0, theta_bull, theta_bear, T, dt, d, rho_bull, rho_bear_1, rho_bear_2, n_bear_periods=10, bear_duration_years=1.0):
    N = int(T/dt)
    delta = np.eye(d)
    mu_bull = theta_bull[0]
    sigma_bull = theta_bull[1]
    mu_bear = theta_bear[0]
    sigma_bear = theta_bear[1]
    
    cov_matrix_bull = sigma_bull**2 * (delta + (1 - delta) * rho_bull)
    
    # We will create two types of bear markets with different correlation structures
    cov_matrix_bear_1 = sigma_bear**2 * (delta + (1 - delta) * rho_bear_1)
    cov_matrix_bear_2 = sigma_bear**2 * (delta + (1 - delta) * rho_bear_2)

    regimes = np.zeros(N, dtype=int)

    #half year period for bear markets is 126*7 observations (7 observations a day and therefore 252*7/2 observations half a year)
    bear_len_steps = int(bear_duration_years *126*7) 
     
    # 2. Randomly assign Bear periods (Non-overlapping)
    bear_starts = []
    max_attempts = 5000 # Prevent infinite loops if too crowded
    attempts = 0

    while len(bear_starts) < n_bear_periods and attempts < max_attempts:
        # Pick a random start index
        start_idx = np.random.randint(0, N - bear_len_steps)
        
        # Check if it overlaps with any existing bear market interval
        overlap = False
        for s in bear_starts:
            if abs(start_idx - s) < bear_len_steps:
                overlap = True
                break
                
        # If no overlap, record it and update the regime mask
        if not overlap:
            bear_starts.append(start_idx)
            #randomly assign this bear period to be either 1 or 2 (the two different bear regimes)
            regimes[start_idx : start_idx + bear_len_steps] = np.random.choice([1, 2])
            
        attempts += 1
        
    if attempts == max_attempts:
        print("Warning: Could not fit all bear periods without overlap. Try reducing duration or number of periods.")

    # 3. Generate Returns (Vectorized for massive speed improvement!)
    drift_bull = (mu_bull - sigma_bull**2 / 2) * dt
    drift_bear = (mu_bear - sigma_bear**2 / 2) * dt

    r_bull = np.random.multivariate_normal(drift_bull * np.ones(d), cov_matrix_bull * dt, size=N)
    r_bear_1 = np.random.multivariate_normal(drift_bear * np.ones(d), cov_matrix_bear_1 * dt, size=N)
    r_bear_2 = np.random.multivariate_normal(drift_bear * np.ones(d), cov_matrix_bear_2 * dt, size=N)
    
    r = np.where(regimes[:, None] == 0, r_bull, np.where(regimes[:, None] == 1, r_bear_1, r_bear_2))
    
    # 4. Construct Prices (Vectorized cumulative sum)
    # S_t = S_0 * exp(sum(r))
    cumulative_returns = np.cumsum(r, axis=0)
    S = S0 * np.exp(cumulative_returns)
    
    t = np.linspace(0, T, N)
    
    # Returning regimes array is helpful for evaluating your k-means accuracy later!
    return t, S, bear_starts, regimes




# make Moons function
def generate_financial_moons(n_samples, d, target_mean, target_cov, noise=0.05):
    """
    Generates a d-dimensional moons dataset with an exact target mean and covariance.
    """
    # 1. Generate base 2D moons
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    # 2. Pad to d dimensions with independent standard normal noise
    extra_dims = d - 2
    X_pad = np.random.normal(0, 1, size=(n_samples, extra_dims))
    X_d = np.hstack((X_2d, X_pad))
    
    # 3. WHITEN THE DATA (Force mean=0, Covariance=Identity)
    emp_mean = np.mean(X_d, axis=0)
    X_centered = X_d - emp_mean
    emp_cov = np.cov(X_centered, rowvar=False)
    
    # Inverse square root of empirical covariance
    inv_sqrt_cov = np.linalg.inv(sqrtm(emp_cov).real) 
    X_white = X_centered.dot(inv_sqrt_cov)
    
    # 4. COLOR THE DATA (Apply target Covariance and Mean)
    # L is the Cholesky lower triangular matrix (target_cov = L @ L.T)
    L = np.linalg.cholesky(target_cov)
    
    # Apply transformation and shift to target mean
    X_final = X_white.dot(L.T) + target_mean
    
    return X_final, y

# (5) Type D Synthetic Data generation function with two parameter theta_bull, theta_bear #nb clusters K = 3
# Regime III has make moons components
def brownian_motion_mult_regime_D(S0, theta_bull, theta_bear, T, dt, d, rho_bull, rho_bear, rho_moon, noise, n_bear_periods=10, bear_duration_years=1.0):
    N = int(T/dt)
    delta = np.eye(d)
    mu_bull = theta_bull[0]
    sigma_bull = theta_bull[1]
    mu_bear = theta_bear[0]
    sigma_bear = theta_bear[1]
    
    cov_matrix_bull = sigma_bull**2 * (delta + (1 - delta) * rho_bull)
    cov_matrix_bear = sigma_bear**2 * (delta + (1 - delta) * rho_bear)
    cov_matrix_moon = sigma_bear**2 * (delta + (1 - delta) * rho_moon)



    regimes = np.zeros(N, dtype=int)

    #half year period for bear markets is 126*7 observations (7 observations a day and therefore 252*7/2 observations half a year)
    bear_len_steps = int(bear_duration_years *126*7) 
     
    # 2. Randomly assign Bear periods (Non-overlapping)
    bear_starts = []
    max_attempts = 5000 # Prevent infinite loops if too crowded
    attempts = 0

    while len(bear_starts) < n_bear_periods and attempts < max_attempts:
        # Pick a random start index
        start_idx = np.random.randint(0, N - bear_len_steps)
        
        # Check if it overlaps with any existing bear market interval
        overlap = False
        for s in bear_starts:
            if abs(start_idx - s) < bear_len_steps:
                overlap = True
                break
                
        # If no overlap, record it and update the regime mask
        if not overlap:
            bear_starts.append(start_idx)
            #randomly assign this bear period to be either 1 or 2 (the two different bear regimes)
            regimes[start_idx : start_idx + bear_len_steps] = np.random.choice([1, 2])
            
        attempts += 1
        
    if attempts == max_attempts:
        print("Warning: Could not fit all bear periods without overlap. Try reducing duration or number of periods.")

    # 3. Generate Returns (Vectorized for massive speed improvement!)
    drift_bull = (mu_bull - sigma_bull**2 / 2) * dt
    drift_bear = (mu_bear - sigma_bear**2 / 2) * dt

    r_bull = np.random.multivariate_normal(drift_bull * np.ones(d), cov_matrix_bull * dt, size=N)
    r_bear = np.random.multivariate_normal(drift_bear * np.ones(d), cov_matrix_bear * dt, size=N)
    r_moon = generate_financial_moons(N, d, target_mean=np.zeros(d)*drift_bear, target_cov=cov_matrix_moon, noise=noise)[0] * np.sqrt(dt) # Scale noise by sqrt(dt) to be consistent with Brownian increments
    
    r = np.where(regimes[:, None] == 0, r_bull, np.where(regimes[:, None] == 1, r_bear, r_moon))
    
    # 4. Construct Prices (Vectorized cumulative sum)
    # S_t = S_0 * exp(sum(r))
    cumulative_returns = np.cumsum(r, axis=0)
    S = S0 * np.exp(cumulative_returns)
    
    t = np.linspace(0, T, N)
    
    # Returning regimes array is helpful for evaluating your k-means accuracy later!
    return t, S, bear_starts, regimes


def visualize_synthetic_data(t, S, bear_starts, K, regimes=None): # K 2 or 3
    d = S.shape[1]
    r_S = np.diff(np.log(S), axis=0)
    # highlight bear periods i
    bear_len_steps = 126*7 # 7 obseervations a day and therefore 252*7 observations a year

    fig, ax = plt.subplots(d+1, 1, figsize=(24, 16))
    ax[0].plot(t, S)
    if K ==2:
        for start in bear_starts:
            ax[0].axvspan(t[start], t[start + bear_len_steps], color='red', alpha=0.3) # Assuming 1 year = 252 trading days
            #label red area as bear market
        ax[0].plot([], [], color='white', label='Regime I')  # Dummy plot for legend
        ax[0].plot([], [], color='red', alpha=0.3, label='Regime II')  # Dummy plot for legend
        ax[0].legend()
        ax[0].set_title("Brownian Motion with Regime I (Bull) and Regime II (Bear) highlighted")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel(f'$\mathbf{{S}}_t$')

        for i in range(1, d+1):
            ax[i].plot(t[1:], r_S[:, i-1])
            ax[i].set_title("Log Returns of Brownian Motion with Regimes")
            ax[i].set_xlabel("t")
            ax[i].set_ylabel(f'$r_t^{{S_{i}}}$')
            for start in bear_starts:
                ax[i].axvspan(t[start], t[start + bear_len_steps], color='red', alpha=0.3) # Assuming 1 year = 252 trading days

        plt.tight_layout()
        plt.show()

    if K ==3:
        for start in bear_starts:
            if regimes[start] == 1: 
                ax[0].axvspan(t[start], t[start + bear_len_steps], color='red', alpha=0.3) # Assuming 1 year = 252 trading days
            elif regimes[start] == 2:
                ax[0].axvspan(t[start], t[start + bear_len_steps], color='green', alpha=0.3) # Assuming 1 year = 252 trading days



        #label white area as bull market
        #label red area as bear market
        ax[0].plot([], [], color='white', label='Regime I')  # Dummy plot for legend
        ax[0].plot([], [], color='red', alpha=0.3, label='Regime II')  # Dummy plot for legend
        ax[0].plot([], [], color='green', alpha=0.3, label='Regime III')  # Dummy plot for legend
        ax[0].legend()
        ax[0].set_title("Brownian Motion with Bull and Bear Regimes")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel(f'$\mathbf{{S}}_t$')


        for i in range(1, d+1):
            ax[i].plot(t[1:], r_S[:, i-1])
            ax[i].set_title("Log Returns of Brownian Motion with Regimes")
            ax[i].set_xlabel("t")
            ax[i].set_ylabel(f'$r_t^{{S_{i}}}$')
            for start in bear_starts:
                if regimes[start] == 1: 
                    ax[i].axvspan(t[start], t[start + bear_len_steps], color='red', alpha=0.3) # Assuming 1 year = 252 trading days
                elif regimes[start] == 2:
                    ax[i].axvspan(t[start], t[start + bear_len_steps], color='green', alpha=0.3) # Assuming 1 year = 252 trading days
        plt.tight_layout()
        plt.show()
      

def visualize_scatter_2D_returns(t,S, K, true_regimes):
    # TODO Implement a function to visualize the 2D scatter plot of the returns colored by the true regimes (only for d=2) to see how well separated the regimes are in the original space before clustering. This will help us understand the difficulty of the clustering task and whether the regimes are easily distinguishable based on their return distributions.
    r_S = np.diff(np.log(S), axis=0)
    _,d  =  r_S.shape
    if d != 2:
        print("Error: This function is only implemented for d=2.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(r_S[true_regimes == 0, 0], r_S[true_regimes == 0, 1], color='blue', alpha=0.5, label='Regime I')
    plt.scatter(r_S[true_regimes == 1, 0], r_S[true_regimes == 1, 1], color='red', alpha=0.5, label='Regime II')
    if K == 3:
        plt.scatter(r_S[true_regimes == 2, 0], r_S[true_regimes == 2, 1], color='green', alpha=0.5, label='Regime III')
    plt.title('2D Scatter Plot of Returns Colored by True Regimes')
    plt.xlabel('$r^{(1)}_{S}$')
    plt.ylabel('$r^{(2)}_{S}$')
    plt.legend()
    plt.show()
    return 

def visualize_scatter_3D_returns(t,S, K, regimes):
    # TODO Implement a function to visualize the 3D scatter plot of the returns colored by the true regimes (only for d=3) to see how well separated the regimes are in the original space before clustering. This will help us understand the difficulty of the clustering task and whether the regimes are easily distinguishable based on their return distributions.
    r_S = np.diff(np.log(S), axis=0)
    _,d  =  r_S.shape
    if d != 3:
        print("Error: This function is only implemented for d=3.")
        return
    from mpl_toolkits.mplot3d import Axes3D 
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r_S[regimes == 0, 0], r_S[regimes == 0, 1], r_S[regimes == 0, 2], color='blue', alpha=0.5, label='Regime I')
    ax.scatter(r_S[regimes == 1, 0], r_S[regimes == 1, 1], r_S[regimes == 1, 2], color='red', alpha=0.5, label='Regime II')
    if K == 3:
        ax.scatter(r_S[regimes == 2, 0], r_S[regimes == 2, 1], r_S[regimes == 2, 2], color='green', alpha=0.5, label='Regime III')
    ax.set_title('3D Scatter Plot of Returns Colored by True Regimes')
    ax.set_xlabel('$r^{(1)}_{S}$')
    ax.set_ylabel('$r^{(2)}_{S}$')
    ax.set_zlabel('$r^{(3)}_{S}$')
    ax.legend()
    plt.show()
    return
    



        
