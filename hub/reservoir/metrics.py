"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Optional, Tuple, Union

import numpy as np


def spectral_radius(weights: np.ndarray) -> float:
    """Compute the spectral radius rho(W) = max_i |lambda_i(W)|.

    Necessary condition for the echo state property in sigmoidal reservoirs
    is rho(W) < 1 (Jaeger, 2002; Verstraeten et al., 2007, Section 2.3).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D matrix.")
    if w.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(w)
    return float(np.max(np.abs(eigenvalues)))


def largest_singular_value(weights: np.ndarray) -> float:
    """Compute the largest singular value sigma_max(W) = ||W||_2.

    Sufficient condition for the echo state property in sigmoidal reservoirs
    is sigma_max(W) < 1 (Jaeger, 2002; Verstraeten et al., 2007, Section 2.3).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("weights must be a 2D matrix.")
    if w.size == 0:
        return 0.0
    svals = np.linalg.svd(w, compute_uv=False)
    return float(svals[0])


def _osborne_balance_log_diag(w: np.ndarray, n_iters: int = 30) -> np.ndarray:
    """Initialize positive diagonal similarity scaling via Osborne balancing."""
    n = w.shape[0]
    d = np.ones(n, dtype=np.float64)
    wd = w.copy()
    for _ in range(n_iters):
        row_norms = np.linalg.norm(wd, axis=1)
        col_norms = np.linalg.norm(wd, axis=0)
        mask = (row_norms > 1e-12) & (col_norms > 1e-12)
        if not np.any(mask):
            break
        scale = np.ones(n, dtype=np.float64)
        scale[mask] = np.sqrt(col_norms[mask] / row_norms[mask])
        d *= scale
        wd = (d[:, None] * w) / d[None, :]
    log_d = np.log(np.maximum(d, 1e-15))
    log_d -= np.mean(log_d)
    return log_d


def _top_singular_triplet(
    matrix: np.ndarray,
    v_init: Optional[np.ndarray] = None,
    power_iters: int = 25,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute largest singular value and left/right singular vectors."""
    if matrix.shape[0] <= 128:
        u_mat, s_vec, vt_mat = np.linalg.svd(matrix, full_matrices=False)
        return float(s_vec[0]), u_mat[:, 0], vt_mat[0, :]
    v = (
        np.ones(matrix.shape[1], dtype=np.float64)
        if v_init is None
        else v_init.astype(np.float64, copy=True)
    )
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        v = np.ones(matrix.shape[1], dtype=np.float64)
        v_norm = np.linalg.norm(v)
    v /= v_norm
    u = np.zeros(matrix.shape[0], dtype=np.float64)
    sigma = 0.0
    for _ in range(power_iters):
        u = matrix @ v
        sigma = float(np.linalg.norm(u))
        if sigma <= 1e-15:
            return 0.0, np.zeros_like(u), v
        u /= sigma
        v = matrix.T @ u
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-15:
            return sigma, u, np.zeros_like(v)
        v /= v_norm
    return sigma, u, v


def structured_singular_value(
    weights: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-5,
    lr: float = 0.5,
) -> float:
    """Compute the D-scaling upper bound for the structured singular value mu(W).

    Implements the tighter sufficient echo state bound from Buehner & Young
    (2006) referenced in Verstraeten et al. (2007, Section 2.3):

        rho(W) <= mu_Delta(W) <= inf_{D > 0, diag} sigma_max(D W D^{-1}) <= sigma_max(W)

    Because f(x) = log sigma_max(exp(diag(x)) W exp(-diag(x))) is convex in x
    with subgradient u^2 - v^2 (where u, v are the leading singular vectors),
    Osborne balancing followed by adaptive subgradient descent finds the tight
    diagonal-scaling upper bound.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be a square 2D matrix.")
    if w.size == 0 or not np.any(w):
        return 0.0

    x = _osborne_balance_log_diag(w)
    d = np.exp(x)
    wd = (d[:, None] * w) / d[None, :]
    best_sigma = largest_singular_value(wd)
    g_sq = np.zeros_like(x)
    v_warm = None

    for _ in range(max_iter):
        d = np.exp(x)
        wd = (d[:, None] * w) / d[None, :]
        sigma, u, v = _top_singular_triplet(wd, v_init=v_warm)
        v_warm = v
        if sigma < best_sigma:
            best_sigma = sigma
        grad = u ** 2 - v ** 2
        if float(np.linalg.norm(grad)) < tol:
            break
        g_sq += grad ** 2
        x = x - (lr / (np.sqrt(g_sq) + 1e-8)) * grad
        x = np.clip(x - np.mean(x), -20.0, 20.0)

    return float(best_sigma)


def rescale_weights(
    weights: np.ndarray,
    target: float,
    metric: str = "spectral_radius",
) -> np.ndarray:
    """Rescale a reservoir weight matrix W so that metric(W) == target.

    Parameters
    ----------
    weights : np.ndarray
        Square reservoir weight matrix.
    target : float
        Desired non-negative metric value.
    metric : str
        One of 'spectral_radius' ('rho'), 'lsv' ('largest_singular_value',
        'sigma'), or 'ssv' ('mu', 'structured_singular_value').
    """
    if target < 0.0:
        raise ValueError("target scale value must be non-negative.")
    w = np.asarray(weights, dtype=np.float64)
    if target == 0.0:
        return np.zeros_like(w)

    key = metric.lower()
    if key in ("spectral_radius", "rho", "sr"):
        current = spectral_radius(w)
    elif key in ("lsv", "largest_singular_value", "sigma"):
        current = largest_singular_value(w)
    elif key in ("ssv", "mu", "structured_singular_value"):
        current = structured_singular_value(w)
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            "Expected 'spectral_radius', 'lsv', or 'ssv'."
        )

    if current <= 1e-15:
        raise ValueError(
            f"Cannot rescale weight matrix with near-zero {metric} ({current})."
        )
    return w * (target / current)


def reservoir_jacobian(
    weights: np.ndarray,
    state: np.ndarray,
    node_type: str = "tanh",
    retainment_rate: float = 0.0,
    pre_activation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the analytical Jacobian Df(s(k)) (Verstraeten et al., 2007, Eqs. 2-3).

    For column-state update s(k+1) = f(s(k)):
      - linear:     J = W
      - tanh:       J = diag(1 - s(k+1)^2) @ W
      - integrator: J = r * I + (1 - r) * diag(1 - tanh(z)^2) @ W
    """
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(state, dtype=np.float64).ravel()
    kind = node_type.lower()

    if kind == "linear":
        return w.copy()
    if kind == "tanh":
        if pre_activation is not None:
            tanh_z = np.tanh(np.asarray(pre_activation, dtype=np.float64).ravel())
        else:
            tanh_z = s
        return (1.0 - tanh_z ** 2)[:, None] * w
    if kind in ("integrator", "analog_integrator", "leaky"):
        r = float(retainment_rate)
        if not 0.0 <= r <= 1.0:
            raise ValueError("retainment_rate must be in [0, 1].")
        if pre_activation is not None:
            tanh_z = np.tanh(np.asarray(pre_activation, dtype=np.float64).ravel())
        else:
            tanh_z = s
        n = w.shape[0]
        return r * np.eye(n, dtype=np.float64) + (1.0 - r) * (
            (1.0 - tanh_z ** 2)[:, None] * w
        )
    raise ValueError(
        f"Analytical Jacobian is not defined for node_type='{node_type}'."
    )


def _step_pre_activation_tanh(
    states: np.ndarray,
    idx: int,
    node_type: str,
    retainment_rate: float,
    pre_activations: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if pre_activations is not None:
        return pre_activations[idx]
    if node_type.lower() in ("integrator", "analog_integrator", "leaky"):
        r = float(retainment_rate)
        if r > 0.0 and r < 1.0 and idx > 0:
            recovered = (states[idx] - r * states[idx - 1]) / (1.0 - r)
            recovered = np.clip(recovered, -0.999999, 0.999999)
            return np.arctanh(recovered)
    return None


def pseudo_lyapunov_exponent(
    weights: np.ndarray,
    states: np.ndarray,
    node_type: str = "tanh",
    retainment_rate: float = 0.0,
    sample_step: int = 10,
    washout: int = 0,
    pre_activations: Optional[np.ndarray] = None,
    return_spectrum: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute the analytical pseudo-Lyapunov exponent h_tilde_max (Section 2.3, Eq. 4).

    At sampled timesteps n (default every 10 timesteps as in Section 2.3), computes
    the singular values r_{n,k} = sqrt(|lambda_k(J_n J_n^T)|) of the local
    analytical Jacobian J_n = Df(s(n)), averages log(r_{n,k}) across sampled
    timesteps (equivalently taking the natural logarithm of the geometric mean
    Lyapunov numbers L_k = prod_n r_{n,k}^{1/M}), and returns
    h_tilde_max = max_k h_tilde_k.

    Parameters
    ----------
    weights : np.ndarray
        Reservoir weight matrix W of shape (N, N).
    states : np.ndarray
        Recorded state trajectory of shape (T, N).
    node_type : str
        'tanh', 'integrator', or 'linear'.
    retainment_rate : float
        Retainment rate r for analog integrator neurons.
    sample_step : int
        Sampling stride across timesteps (paper uses 10).
    washout : int
        Number of initial transient timesteps to ignore.
    pre_activations : np.ndarray, optional
        Optional pre-activation array of shape (T, N) for exact integrator Jacobians.
    return_spectrum : bool
        If True, returns (h_max, spectrum_array).
    """
    states_arr = np.asarray(states, dtype=np.float64)
    if states_arr.ndim != 2:
        raise ValueError("states must be a 2D array of shape (T, N).")
    if sample_step < 1:
        raise ValueError("sample_step must be >= 1.")
    if washout < 0 or washout >= states_arr.shape[0]:
        raise ValueError("washout must be in [0, T - 1].")

    indices = range(washout, states_arr.shape[0], sample_step)
    log_sv_sum = np.zeros(states_arr.shape[1], dtype=np.float64)
    count = 0
    for idx in indices:
        pre = _step_pre_activation_tanh(
            states_arr, idx, node_type, retainment_rate, pre_activations
        )
        jac = reservoir_jacobian(
            weights,
            states_arr[idx],
            node_type=node_type,
            retainment_rate=retainment_rate,
            pre_activation=pre,
        )
        svals = np.linalg.svd(jac, compute_uv=False)
        log_sv_sum += np.log(np.maximum(svals, 1e-15))
        count += 1

    spectrum = log_sv_sum / max(count, 1)
    h_max = float(np.max(spectrum))
    if return_spectrum:
        return h_max, spectrum
    return h_max


def max_lyapunov_exponent(
    weights: np.ndarray,
    states: np.ndarray,
    node_type: str = "tanh",
    retainment_rate: float = 0.0,
    washout: int = 0,
    pre_activations: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute the finite-time maximum Lyapunov exponent via tangent propagation.

    Unlike the local single-step pseudo-Lyapunov exponent in Eq. (4), this
    propagates a unit tangent vector v_{t+1} = J_t v_t with renormalization
    (Benettin method) to measure asymptotic trajectory divergence (h > 0 indicates
    chaotic sensitivity).
    """
    states_arr = np.asarray(states, dtype=np.float64)
    if states_arr.ndim != 2:
        raise ValueError("states must be a 2D array of shape (T, N).")
    if washout < 0 or washout >= states_arr.shape[0]:
        raise ValueError("washout must be in [0, T - 1].")

    gen = np.random.default_rng(0) if rng is None else rng
    v = gen.standard_normal(states_arr.shape[1])
    v /= np.linalg.norm(v)

    log_growth = 0.0
    count = 0
    for idx in range(washout, states_arr.shape[0]):
        pre = _step_pre_activation_tanh(
            states_arr, idx, node_type, retainment_rate, pre_activations
        )
        jac = reservoir_jacobian(
            weights,
            states_arr[idx],
            node_type=node_type,
            retainment_rate=retainment_rate,
            pre_activation=pre,
        )
        v = jac @ v
        norm = float(np.linalg.norm(v))
        if norm <= 1e-15:
            log_growth += np.log(1e-15)
            v = gen.standard_normal(states_arr.shape[1])
            v /= np.linalg.norm(v)
        else:
            log_growth += np.log(norm)
            v /= norm
        count += 1
    return float(log_growth / max(count, 1))


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean squared error (Verstraeten et al., 2007, Appendix A.1)."""
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((pred - true) ** 2))


def nrmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute normalized root mean squared error (RMSE / std(y_true))."""
    true = np.asarray(y_true, dtype=np.float64)
    std = float(np.std(true))
    if std <= 1e-15:
        return float(np.sqrt(mse(y_pred, true)))
    return float(np.sqrt(mse(y_pred, true)) / std)


def memory_capacity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    return_per_delay: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute Jaeger's memory capacity MC = sum_k MC_k (Appendix A.2).

    Each k-delay memory capacity is the squared correlation coefficient:
        MC_k = cov^2(s_k, y_k) / (var(s_k) * var(y_k)).
    """
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    if true.ndim == 1:
        true = true[:, None]
    if pred.shape != true.shape:
        raise ValueError(
            f"y_pred shape {pred.shape} does not match y_true shape {true.shape}."
        )

    pred_c = pred - np.mean(pred, axis=0, keepdims=True)
    true_c = true - np.mean(true, axis=0, keepdims=True)
    cov = np.mean(pred_c * true_c, axis=0)
    var_p = np.mean(pred_c ** 2, axis=0)
    var_t = np.mean(true_c ** 2, axis=0)
    denom = var_p * var_t
    mc_k = np.where(denom > 1e-15, (cov ** 2) / np.maximum(denom, 1e-15), 0.0)
    mc_k = np.clip(mc_k, 0.0, 1.0)
    total = float(np.sum(mc_k))
    if return_per_delay:
        return total, mc_k
    return total


def word_error_rate(y_pred_labels: np.ndarray, y_true_labels: np.ndarray) -> float:
    """Compute classification Word Error Rate (fraction of misclassified samples)."""
    pred = np.asarray(y_pred_labels).ravel()
    true = np.asarray(y_true_labels).ravel()
    if pred.shape != true.shape:
        raise ValueError("y_pred_labels and y_true_labels must have the same length.")
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred != true))
