"""
Ledoit & Wolf (2008) — Robust Performance Hypothesis Testing with the Sharpe Ratio
====================================================================================
Implements:
  1) HAC inference (prewhitened QS kernel, Andrews-Monahan 1992)
  2) Studentized circular block bootstrap with data-driven block-size calibration

Reference:
  Ledoit, O. & Wolf, M. (2008). "Robust performance hypothesis testing with the
  Sharpe ratio." Journal of Empirical Finance, 15(5), 850–859.

Usage:
  Call  ledoit_wolf_test(returns_i, returns_n, ...)  where
    - returns_i : 1-D array of excess returns for strategy i
    - returns_n : 1-D array of excess returns for the benchmark (strategy n)
  Both arrays must have the same length T and be aligned in time.
"""

import numpy as np
from scipy.stats import norm


# =============================================================================
# Core building blocks
# =============================================================================

def _sharpe(mu, gamma):
    """Sharpe ratio from mean and raw second moment: mu / sqrt(gamma - mu^2)."""
    var = gamma - mu ** 2
    if var <= 0:
        return 0.0
    return mu / np.sqrt(var)


def _delta_sharpe(v):
    """Δ = SR_i − SR_n  from the 4-vector v = (mu_i, mu_n, gamma_i, gamma_n)."""
    return _sharpe(v[0], v[2]) - _sharpe(v[1], v[3])


def _gradient(v):
    """∇f  evaluated at v = (mu_i, mu_n, gamma_i, gamma_n).
    Equation from Section 3 of Ledoit & Wolf (2008)."""
    mu_i, mu_n, g_i, g_n = v
    var_i = g_i - mu_i ** 2
    var_n = g_n - mu_n ** 2
    eps = 1e-14
    var_i = max(var_i, eps)
    var_n = max(var_n, eps)
    return np.array([
         g_i / (var_i ** 1.5),
        -g_n / (var_n ** 1.5),
        -0.5 * mu_i / (var_i ** 1.5),
         0.5 * mu_n / (var_n ** 1.5),
    ])


def _v_hat(ri, rn):
    """Estimate v = (mu_i, mu_n, gamma_i, gamma_n) from return vectors."""
    return np.array([ri.mean(), rn.mean(), (ri ** 2).mean(), (rn ** 2).mean()])


def _y_matrix(ri, rn, v):
    """T × 4  matrix of centred influence vectors y_t."""
    T = len(ri)
    Y = np.column_stack([
        ri - v[0],
        rn - v[1],
        ri ** 2 - v[2],
        rn ** 2 - v[3],
    ])
    return Y


# =============================================================================
# HAC kernel estimation  (prewhitened Quadratic-Spectral, Andrews & Monahan 1992)
# =============================================================================

def _qs_kernel(x):
    """Quadratic-Spectral kernel."""
    if np.abs(x) < 1e-8:
        return 1.0
    z = 6.0 * np.pi * x / 5.0
    return 25.0 / (12.0 * np.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))


def _auto_bandwidth_qs(Y):
    """Andrews (1991) automatic bandwidth for the QS kernel.
    Uses AR(1) approximation per column to estimate alpha(2)."""
    T, d = Y.shape
    alphas_num = 0.0
    alphas_den = 0.0
    for j in range(d):
        y = Y[:, j]
        # fit AR(1)
        rho = np.corrcoef(y[:-1], y[1:])[0, 1]
        rho = np.clip(rho, -0.99, 0.99)
        sig2 = np.var(y) * (1 - rho ** 2)
        alphas_num += 4 * rho ** 2 * sig2 ** 2 / ((1 - rho) ** 8)
        alphas_den += sig2 ** 2 / ((1 - rho) ** 4)
    if alphas_den == 0:
        return 1.0
    alpha2 = alphas_num / alphas_den
    ST = 1.3221 * (alpha2 * T) ** 0.2
    return max(ST, 1.0)


def _prewhiten(Y):
    """VAR(1) prewhitening: returns whitened residuals and the AR coefficient matrix."""
    T, d = Y.shape
    X = Y[:-1]  # T-1 × d
    Yp = Y[1:]  # T-1 × d
    # OLS:  A = (X'X)^{-1} X'Y
    A = np.linalg.lstsq(X, Yp, rcond=None)[0]  # d × d
    resid = Yp - X @ A
    return resid, A


def _hac_psi(Y, prewhiten=True):
    """HAC estimate of Ψ using QS kernel (optionally prewhitened)."""
    T, d = Y.shape
    if prewhiten:
        resid, A = _prewhiten(Y)
        S_T = _auto_bandwidth_qs(resid)
        T2 = resid.shape[0]
        # kernel estimate on residuals
        Gamma0 = resid.T @ resid / T2
        Psi_resid = Gamma0.copy()
        for j in range(1, T2):
            w = _qs_kernel(j / S_T)
            if abs(w) < 1e-12:
                continue
            Gj = resid[j:].T @ resid[:-j] / T2
            Psi_resid += w * (Gj + Gj.T)
        # recolour:  Ψ = (I - A)^{-1} Ψ_resid (I - A')^{-1}
        Ainv = np.linalg.inv(np.eye(d) - A)
        Psi = Ainv @ Psi_resid @ Ainv.T
    else:
        S_T = _auto_bandwidth_qs(Y)
        Gamma0 = Y.T @ Y / T
        Psi = Gamma0.copy()
        for j in range(1, T):
            w = _qs_kernel(j / S_T)
            if abs(w) < 1e-12:
                continue
            Gj = Y[j:].T @ Y[:-j] / T
            Psi += w * (Gj + Gj.T)
    # degrees-of-freedom correction  T / (T - 4)
    Psi *= T / (T - 4)
    return Psi


def _se_delta(v, Psi, T):
    """Standard error of Δ̂  via the delta method: sqrt( ∇f' Ψ ∇f / T )."""
    g = _gradient(v)
    var = g @ Psi @ g / T
    return np.sqrt(max(var, 0.0))


# =============================================================================
# Studentized circular block bootstrap  (Section 3.2.2)
# =============================================================================

def _circular_block_resample(data, b):
    """Circular block bootstrap resample of a T × d array with block size b."""
    T, d = data.shape
    n_blocks = int(np.ceil(T / b))
    indices = []
    for _ in range(n_blocks):
        start = np.random.randint(0, T)
        block = [(start + j) % T for j in range(b)]
        indices.extend(block)
    return data[np.array(indices[:T])]


def _bootstrap_se(ri_star, rn_star, b):
    """Bootstrap standard error using the block-variance estimator of
    Götze & Künsch (1996), as described in Section 3.2.2."""
    T = len(ri_star)
    v_star = _v_hat(ri_star, rn_star)
    Y_star = _y_matrix(ri_star, rn_star, v_star)
    l = T // b
    if l < 2:
        # fallback to HAC on bootstrap sample
        Psi_star = _hac_psi(Y_star, prewhiten=False)
        return _se_delta(v_star, Psi_star, T)
    # block means
    phi = np.zeros((l, 4))
    for j in range(l):
        phi[j] = Y_star[j * b:(j + 1) * b].sum(axis=0) / np.sqrt(b)
    Psi_star = phi.T @ phi / l
    return _se_delta(v_star, Psi_star, T)


# =============================================================================
# Block-size calibration  (Algorithm 3.1)
# =============================================================================

def _fit_var1_and_resample(ri, rn, T_out, rng):
    """Fit VAR(1) to (ri, rn) and generate a pseudo sequence of length T_out
    using the stationary bootstrap (avg block size = 5) on residuals."""
    data = np.column_stack([ri, rn])
    T = len(ri)
    # VAR(1) OLS
    X = data[:-1]
    Yp = data[1:]
    mu = data.mean(axis=0)
    Xc = X - mu
    Ypc = Yp - mu
    A = np.linalg.lstsq(Xc, Ypc, rcond=None)[0]
    resid = Ypc - Xc @ A  # (T-1) × 2

    # stationary bootstrap of residuals (geometric block, avg = 5)
    p = 1.0 / 5.0
    T_res = len(resid)
    idx = np.zeros(T_out, dtype=int)
    idx[0] = rng.integers(0, T_res)
    for t in range(1, T_out):
        if rng.random() < p:
            idx[t] = rng.integers(0, T_res)
        else:
            idx[t] = (idx[t - 1] + 1) % T_res
    boot_resid = resid[idx]

    # reconstruct series
    sim = np.zeros((T_out, 2))
    sim[0] = mu + boot_resid[0]
    for t in range(1, T_out):
        sim[t] = mu + A.T @ (sim[t - 1] - mu) + boot_resid[t]
    return sim[:, 0], sim[:, 1]


def _calibrate_block_size(ri, rn, block_sizes, alpha=0.05, K=500, M_cal=499, seed=42):
    """Algorithm 3.1 — estimate the coverage function g(b) and return the
    block size closest to nominal 1-α coverage."""
    rng = np.random.default_rng(seed)
    T = len(ri)
    delta_hat = _delta_sharpe(_v_hat(ri, rn))
    coverage = {}

    for b in block_sizes:
        hits = 0
        for k in range(K):
            ri_pseudo, rn_pseudo = _fit_var1_and_resample(ri, rn, T, rng)
            v_pseudo = _v_hat(ri_pseudo, rn_pseudo)
            delta_pseudo = _delta_sharpe(v_pseudo)
            Y_pseudo = _y_matrix(ri_pseudo, rn_pseudo, v_pseudo)
            Psi_pseudo = _hac_psi(Y_pseudo, prewhiten=True)
            se_pseudo = _se_delta(v_pseudo, Psi_pseudo, T)
            if se_pseudo < 1e-14:
                continue
            d_orig = abs(delta_pseudo) / se_pseudo

            # bootstrap world
            data_pseudo = np.column_stack([ri_pseudo, rn_pseudo])
            count_ge = 0
            for m in range(M_cal):
                star = _circular_block_resample(data_pseudo, b)
                ri_s, rn_s = star[:, 0], star[:, 1]
                v_s = _v_hat(ri_s, rn_s)
                delta_s = _delta_sharpe(v_s)
                se_s = _bootstrap_se(ri_s, rn_s, b)
                if se_s < 1e-14:
                    continue
                d_star = abs(delta_s - delta_pseudo) / se_s
                if d_star >= d_orig:
                    count_ge += 1
            pv = (count_ge + 1) / (M_cal + 1)
            if pv > alpha:
                hits += 1
        coverage[b] = hits / K
        print(f"  block size b={b:>2d}  →  coverage = {coverage[b]:.4f}  (target = {1 - alpha:.2f})")

    # pick b minimising |coverage - (1-alpha)|
    best_b = min(block_sizes, key=lambda b: abs(coverage[b] - (1 - alpha)))
    return best_b, coverage


# =============================================================================
# Public API
# =============================================================================

def ledoit_wolf_test(returns_i, returns_n,
                     method="both",
                     n_bootstrap=4999,
                     block_sizes=None,
                     calibrate=True,
                     alpha=0.05,
                     K_calibration=1000,
                     M_calibration=499,
                     seed=42):
    """
    Ledoit & Wolf (2008) two-sided test for H0: SR_i = SR_n.

    Parameters
    ----------
    returns_i, returns_n : 1-D array-like
        Excess returns (over the risk-free rate) of equal length T.
    method : str
        'hac'  — HAC inference only
        'boot' — studentized circular block bootstrap only
        'both' — run both (default)
    n_bootstrap : int
        Number of bootstrap resamples M (default 4999).
    block_sizes : list[int] or None
        Candidate block sizes for calibration (default [1,2,4,6,8,10]).
    calibrate : bool
        If True, run Algorithm 3.1 to pick the block size.
        If False, use the largest block size in `block_sizes`.
    alpha : float
        Significance level (default 0.05).
    K_calibration : int
        Number of pseudo sequences in Algorithm 3.1 (default 1000).
    M_calibration : int
        Bootstrap resamples inside each calibration run (default 499).
    seed : int
        Random seed.

    Returns
    -------
    results : dict with keys
        'SR_i', 'SR_n'        — estimated Sharpe ratios
        'delta'               — SR_i − SR_n
        'hac_se', 'hac_pval'  — HAC standard error & two-sided p-value
        'boot_pval'           — bootstrap p-value
        'boot_block_size'     — chosen block size
        'boot_ci_lower/upper' — bootstrap 1−α confidence interval
    """
    ri = np.asarray(returns_i, dtype=float)
    rn = np.asarray(returns_n, dtype=float)
    assert len(ri) == len(rn), "Return series must have equal length."
    T = len(ri)

    v = _v_hat(ri, rn)
    delta = _delta_sharpe(v)
    sr_i = _sharpe(v[0], v[2])
    sr_n = _sharpe(v[1], v[3])

    results = {
        "SR_i": sr_i,
        "SR_n": sr_n,
        "delta": delta,
        "T": T,
    }

    # --- HAC inference ---
    if method in ("hac", "both"):
        Y = _y_matrix(ri, rn, v)
        Psi = _hac_psi(Y, prewhiten=True)
        se = _se_delta(v, Psi, T)
        if se > 0:
            z = abs(delta) / se
            pval = 2 * norm.sf(z)
        else:
            pval = np.nan
        results["hac_se"] = se
        results["hac_pval"] = pval

    # --- Bootstrap inference ---
    if method in ("boot", "both"):
        if block_sizes is None:
            block_sizes = [1, 2, 4, 6, 8, 10]

        if calibrate:
            print("Running block-size calibration (this may take a while)...")
            best_b, cov_fn = _calibrate_block_size(
                ri, rn, block_sizes, alpha=alpha,
                K=K_calibration, M_cal=M_calibration, seed=seed
            )
            print(f"Optimal block size: b = {best_b}")
            results["calibration_coverage"] = cov_fn
        else:
            best_b = max(block_sizes)

        results["boot_block_size"] = best_b

        # Original studentized statistic
        Y = _y_matrix(ri, rn, v)
        Psi = _hac_psi(Y, prewhiten=True)
        se_orig = _se_delta(v, Psi, T)
        d_orig = abs(delta) / se_orig if se_orig > 0 else 0.0

        data = np.column_stack([ri, rn])
        rng = np.random.default_rng(seed + 1)

        d_stars = np.zeros(n_bootstrap)
        delta_stars = np.zeros(n_bootstrap)
        se_stars = np.zeros(n_bootstrap)

        for m in range(n_bootstrap):
            np.random.seed(rng.integers(0, 2 ** 31))
            star = _circular_block_resample(data, best_b)
            ri_s, rn_s = star[:, 0], star[:, 1]
            v_s = _v_hat(ri_s, rn_s)
            delta_s = _delta_sharpe(v_s)
            se_s = _bootstrap_se(ri_s, rn_s, best_b)
            delta_stars[m] = delta_s
            se_stars[m] = se_s
            if se_s > 1e-14:
                d_stars[m] = abs(delta_s - delta) / se_s
            else:
                d_stars[m] = 0.0

        # p-value (Remark 3.2, Eq. 9)
        pval_boot = (np.sum(d_stars >= d_orig) + 1) / (n_bootstrap + 1)
        results["boot_pval"] = pval_boot

        # Confidence interval (Eq. 7)
        q = np.quantile(d_stars, 1 - alpha)
        results["boot_ci_lower"] = delta - q * se_orig
        results["boot_ci_upper"] = delta + q * se_orig

    return results


def print_results(results, strategy_name="Strategy", benchmark_name="Benchmark"):
    """Pretty-print the test results."""
    print("=" * 65)
    print("  Ledoit & Wolf (2008) Sharpe Ratio Difference Test")
    print("=" * 65)
    print(f"  {strategy_name} SR  :  {results['SR_i']:.4f}")
    print(f"  {benchmark_name} SR :  {results['SR_n']:.4f}")
    print(f"  Δ (SR_i − SR_n)    :  {results['delta']:.4f}")
    print(f"  T (observations)   :  {results['T']}")
    print("-" * 65)

    if "hac_pval" in results:
        print(f"  HAC p-value        :  {results['hac_pval']:.4f}")
        print(f"  HAC std error      :  {results['hac_se']:.6f}")
    if "boot_pval" in results:
        print(f"  Bootstrap p-value  :  {results['boot_pval']:.4f}")
        print(f"  Block size (b)     :  {results['boot_block_size']}")
        print(f"  95% CI             :  [{results['boot_ci_lower']:.4f}, {results['boot_ci_upper']:.4f}]")
    print("=" * 65)
    # Interpretation
    alpha = 0.05
    key = "boot_pval" if "boot_pval" in results else "hac_pval"
    if key in results:
        if results[key] < alpha:
            print(f"  ➜  REJECT H0 at {alpha:.0%}: the Sharpe ratios are significantly different.")
        else:
            print(f"  ➜  FAIL TO REJECT H0 at {alpha:.0%}: no significant difference in Sharpe ratios.")
    print()


# =============================================================================
# Integration with your trading strategies
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Replace the dummy data below with your actual portfolio values.
    # They should already be defined in your notebook / script, e.g.:
    #
    #   portfolio_value_unif                       — from long_strat_unifortho
    #   portfolio_value_unif_implied_conviction     — from long_strat_implied (conviction)
    #   portfolio_value_unif_implied_continuous      — from long_strat_implied (continuous)
    #   portfolio_value_unif_implied_hysteresis      — from long_strat_implied (hysteresis)
    #   portfolio_value_long                        — from long_only (benchmark)
    #
    # All portfolio_value arrays must start from the same date and have
    # the same length.  Trim them to their common date range first.
    # ------------------------------------------------------------------

    # -------- EXAMPLE with random data (replace with yours) -----------
    np.random.seed(0)
    T = 252 * 14  # ~14 years of daily data
    portfolio_value_unif = 10000 * np.cumprod(1 + np.random.normal(0.0003, 0.008, T))
    portfolio_value_unif_implied_conviction = 10000 * np.cumprod(1 + np.random.normal(0.0002, 0.007, T))
    portfolio_value_unif_implied_continuous = 10000 * np.cumprod(1 + np.random.normal(0.00025, 0.009, T))
    portfolio_value_unif_implied_hysteresis = 10000 * np.cumprod(1 + np.random.normal(0.00028, 0.0075, T))
    portfolio_value_long = 10000 * np.cumprod(1 + np.random.normal(0.00015, 0.01, T))

    # ------------------------------------------------------------------
    # Helper: portfolio values  →  daily returns
    # ------------------------------------------------------------------
    def pv_to_returns(pv):
        pv = np.asarray(pv, dtype=float)
        return np.diff(pv) / pv[:-1]

    benchmark_returns = pv_to_returns(portfolio_value_long)

    strategies = {
        "sWk-Means (unifortho)": portfolio_value_unif,
        "sWk-Means Implied Conviction": portfolio_value_unif_implied_conviction,
        "sWk-Means Implied Continuous": portfolio_value_unif_implied_continuous,
        "sWk-Means Implied Hysteresis": portfolio_value_unif_implied_hysteresis,
    }

    for name, pv in strategies.items():
        strat_returns = pv_to_returns(pv)
        # Ensure equal length (trim to shorter)
        min_len = min(len(strat_returns), len(benchmark_returns))
        ri = strat_returns[:min_len]
        rn = benchmark_returns[:min_len]

        print(f"\n{'#' * 65}")
        print(f"#  Testing: {name}  vs  Long-Only Benchmark")
        print(f"{'#' * 65}\n")

        # For a quick run use method="hac" (instant).
        # For the full robust test use method="both" (slow but recommended).
        res = ledoit_wolf_test(
            ri, rn,
            method="hac",          # change to "both" for bootstrap
            n_bootstrap=4999,
            calibrate=True,        # Algorithm 3.1 for block size
            K_calibration=500,     # increase to 1000+ for production
            M_calibration=299,     # increase to 499+ for production
            alpha=0.05,
            seed=42,
        )
        print_results(res, strategy_name=name, benchmark_name="Long-Only")
