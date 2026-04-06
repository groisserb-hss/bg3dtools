"""
Statistical analysis utilities.

This module provides functions for computing intraclass correlation
coefficients (ICC), confidence intervals, and permutation tests.
"""

from itertools import product
from typing import Optional, Tuple, Union
import numpy as np



def icc_two_way_ms(X: np.ndarray) -> Tuple[float, float, float, int, int]:
    """
    Compute two-way ANOVA mean squares (balanced) for an N×K ratings matrix.
    Returns MSR (subjects), MSC (raters), MSE (residual), along with N, K.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D: shape (n_subjects, n_raters).")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains NaN/inf; provide complete data.")
    N, K = X.shape
    if N < 2 or K < 2:
        raise ValueError("Need at least 2 subjects and 2 raters to compute ICC.")

    grand = X.mean()
    row_means = X.mean(axis=1, keepdims=True)
    col_means = X.mean(axis=0, keepdims=True)

    # Sums of squares
    ss_between_subjects = K * np.sum((row_means - grand)**2)
    ss_between_raters   = N * np.sum((col_means - grand)**2)
    ss_total            = np.sum((X - grand)**2)
    ss_error            = ss_total - ss_between_subjects - ss_between_raters

    # Mean squares
    MSR = ss_between_subjects / (N - 1)            # subjects
    MSC = ss_between_raters   / (K - 1)            # raters
    MSE = ss_error            / ((N - 1) * (K - 1))# residual
    return MSR, MSC, MSE, N, K


def _icc_point(MSR: float, MSC: float, MSE: float, N: int, K: int, icc_type: str = "ICC2") -> float:
    icc_type = icc_type.upper()
    if icc_type == "ICC2":     # two-way random, absolute agreement  (ICC(A,1))
        denom = MSR + (K - 1)*MSE + (K * (MSC - MSE) / N)
        return (MSR - MSE) / denom
    elif icc_type == "ICC3":   # two-way mixed, consistency           (ICC(C,1))
        denom = MSR + (K - 1)*MSE
        return (MSR - MSE) / denom
    else:
        raise ValueError("icc_type must be 'ICC2' or 'ICC3'.")


def icc_value(
    X: np.ndarray,
    icc_type: str = "ICC2",
    alpha: float = 0.05,
    return_ci: bool = False,
    n_boot: int = 5000,
    random_state: int | None = 42,
) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
    Point estimate and 95% CI for ICC(2,1) or ICC(3,1).

    CI is computed via *parametric bootstrap* of a two-way model with a
    Fisher r→z transform (atanh/tanh) for numerical stability near ICC≈1.
      - ICC(2,1): random SUBJECT and random RATER effects.
      - ICC(3,1): random SUBJECT effects + fixed RATER effects (kept as observed).

    Parameters
    ----------
    X : (N, K) array
        Ratings matrix: rows=subjects, cols=raters (or trials).
    icc_type : {"ICC2", "ICC3"}
    alpha : float
        1 - confidence level (default 0.05 for 95% CI).
    n_boot : int
        Parametric bootstrap draws.
    random_state : int | None
        RNG seed.

    Returns
    -------
    icc : float
        Point estimate.
    ci_low, ci_high : float
        Lower/upper bounds of the (1-alpha) CI.
    """
    X = np.asarray(X, dtype=float)
    MSR, MSC, MSE, N, K = icc_two_way_ms(X)
    icc_hat = _icc_point(MSR, MSC, MSE, N, K, icc_type=icc_type)

    if return_ci:
        rng = np.random.default_rng(random_state)
        grand = X.mean()
        row_means = X.mean(axis=1)
        col_means = X.mean(axis=0)
        rater_fixed = col_means - grand  # for ICC3 (mixed), keep observed fixed effects

        # Variance components (non-negative) for parametric bootstrap
        # E[MSR] = K*sigma_s^2 + sigma_e^2
        # E[MSC] = N*sigma_r^2 + sigma_e^2
        # E[MSE] = sigma_e^2
        sigma_e2 = max(MSE, 0.0)
        sigma_s2 = max((MSR - MSE) / K, 0.0)
        sigma_r2 = max((MSC - MSE) / N, 0.0)  # used only for ICC2

        # Vectorized bootstrap: generate all (n_boot, N, K) datasets at once
        s_all = rng.normal(0.0, np.sqrt(sigma_s2), size=(n_boot, N))  # (n_boot, N)

        if icc_type.upper() == "ICC2":
            r_all = rng.normal(0.0, np.sqrt(sigma_r2), size=(n_boot, K))  # (n_boot, K)
        else:
            r_all = np.broadcast_to(rater_fixed, (n_boot, K))

        e_all = rng.normal(0.0, np.sqrt(sigma_e2), size=(n_boot, N, K))
        Xb_all = grand + s_all[:, :, None] + r_all[:, None, :] + e_all  # (n_boot, N, K)

        # Compute ANOVA mean squares for all bootstrap samples in parallel
        grand_b = Xb_all.mean(axis=(1, 2))  # (n_boot,)
        row_means_b = Xb_all.mean(axis=2)   # (n_boot, N)
        col_means_b = Xb_all.mean(axis=1)   # (n_boot, K)

        ss_subj = K * np.sum((row_means_b - grand_b[:, None])**2, axis=1)
        ss_rater = N * np.sum((col_means_b - grand_b[:, None])**2, axis=1)
        ss_total = np.sum((Xb_all - grand_b[:, None, None])**2, axis=(1, 2))
        ss_error = ss_total - ss_subj - ss_rater

        MSR_b = ss_subj / (N - 1)
        MSC_b = ss_rater / (K - 1)
        MSE_b = ss_error / ((N - 1) * (K - 1))

        # Vectorized ICC point estimates
        icc_upper = icc_type.upper()
        if icc_upper == "ICC2":
            denom = MSR_b + (K - 1)*MSE_b + (K * (MSC_b - MSE_b) / N)
            iccs = (MSR_b - MSE_b) / denom
        else:  # ICC3
            denom = MSR_b + (K - 1)*MSE_b
            iccs = (MSR_b - MSE_b) / denom

        # Fisher z-transform for stable CI near boundaries
        eps = 1e-12
        # z_hat = np.arctanh(np.clip(icc_hat, -1 + eps, 1 - eps))
        z_samp = np.arctanh(np.clip(iccs,   -1 + eps, 1 - eps))

        lo_q = 100 * (alpha / 2.0)
        hi_q = 100 * (1.0 - alpha / 2.0)

        # Percentile CI on z-scale, then invert
        z_lo = np.percentile(z_samp, lo_q)
        z_hi = np.percentile(z_samp, hi_q)
        ci_low = np.tanh(z_lo)
        ci_high = np.tanh(z_hi)

        return float(icc_hat), (float(ci_low), float(ci_high))
    else:
        return float(icc_hat)


def perm_test_icc_diff_arrays(
    A: np.ndarray,
    B: np.ndarray,
    icc_type: str = "ICC2",
    n_perm: int = 10000,
    random_state: int | None = 42,
    exhaustive: bool = False,
    max_exhaustive_configs: int = 10_000_000,
):
    """
    Permutation test for ΔICC = ICC(A) - ICC(B) with paired N×K arrays.

    Swaps condition labels **per subject (row-wise)** to preserve within-subject
    correlation across raters. Two-sided p-value.

    Parameters
    ----------
    A, B : (N, K) arrays
        Paired measurements for two conditions (e.g., ST vs Spirometry).
        Row i = same subject in both A and B; columns = same raters/trials.
        No NaNs; balanced K across conditions.
    icc_type : {"ICC2","ICC3"}
    n_perm : int
        Monte Carlo permutations if exhaustive=False.
    random_state : int or None
        RNG seed for Monte Carlo mode.
    exhaustive : bool
        If True, enumerate ALL 2^N subject-level swap patterns (exact test).
    max_exhaustive_configs : int
        Safety cap on total configurations.

    Returns
    -------
    dict with:
      - icc_A, icc_B, diff_obs
      - p_value   (exact if exhaustive=True; Monte Carlo otherwise)
      - diffs     (array of permutation diffs)
      - mode      ("exhaustive" or "monte_carlo")
      - n_evaluated, n_subjects, n_raters, icc_type
    """
    from tqdm import tqdm

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError("A and B must be 2D and of identical shape (N×K).")
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(B)):
        raise ValueError("A or B contains NaN/inf; provide complete data.")

    N, K = A.shape
    icc_A = icc_value(A, icc_type=icc_type)
    icc_B = icc_value(B, icc_type=icc_type)
    diff_obs = icc_A - icc_B

    def _swap_rows(Ain, Bin, swap_mask_rows):
        """Swap whole rows where swap_mask_rows[i] is True."""
        Aout = Ain.copy()
        Bout = Bin.copy()
        idx = np.where(swap_mask_rows)[0]
        if idx.size:
            Aout[idx, :], Bout[idx, :] = Bin[idx, :], Ain[idx, :]
        return Aout, Bout

    if exhaustive:
        total = 1 << N  # 2^N row-swap patterns
        if total > max_exhaustive_configs:
            raise ValueError(
                f"Exhaustive mode would require {total:,} configurations; "
                f"reduce N or increase max_exhaustive_configs, or use Monte Carlo."
            )
        diffs = np.empty(total, dtype=float)
        for i, bits in tqdm(enumerate(product((0, 1), repeat=N))):
            swap_rows = np.fromiter(bits, dtype=bool, count=N)
            permA, permB = _swap_rows(A, B, swap_rows)
            diffs[i] = icc_value(permA, icc_type=icc_type) - icc_value(permB, icc_type=icc_type)
        p_val = np.mean(np.abs(diffs) >= np.abs(diff_obs))  # exact two-sided
        mode = "exhaustive"
        n_eval = total
    else:
        rng = np.random.default_rng(random_state)
        diffs = np.empty(n_perm, dtype=float)
        for i in tqdm(range(n_perm)):
            swap_rows = rng.random(N) < 0.5
            permA, permB = _swap_rows(A, B, swap_rows)
            diffs[i] = icc_value(permA, icc_type=icc_type) - icc_value(permB, icc_type=icc_type)
        # Monte Carlo two-sided p with +1 correction
        p_val = (np.sum(np.abs(diffs) >= np.abs(diff_obs)) + 1.0) / (n_perm + 1.0)
        mode = "monte_carlo"
        n_eval = n_perm

    return {
        "icc_type": icc_type.upper(),
        "n_subjects": int(N),
        "n_raters": int(K),
        "icc_A": float(icc_A),
        "icc_B": float(icc_B),
        "diff_obs": float(diff_obs),
        "p_value": float(p_val),
        "diffs": diffs,
        "mode": mode,
        "n_evaluated": int(n_eval),
    }


