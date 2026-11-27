"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import List, Optional, Tuple, Union

import numpy as np


def _get_rng(
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def generate_narma10(
    n_steps: int = 2000,
    u_range: Tuple[float, float] = (0.0, 0.5),
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the 10th-order NARMA benchmark (Section 2 & Appendix A.1).

    System equation (Jaeger, 2003; Steil, 2005; Verstraeten et al., 2007):

        y(k + 1) = 0.3 y(k) + 0.05 y(k) * sum_{i=0}^{9} y(k - i)
                   + 1.5 u(k - 9) u(k) + 0.1

    Note: Appendix A.1 prints u(k) in [0, 1], which is a known typo in the
    paper because the bilinear term causes y(k) to diverge to infinity on
    [0, 1]. The standard stable range from Jaeger (2003) is [0.0, 0.5]
    (default here).

    Returns
    -------
    u : np.ndarray of shape (n_steps, 1)
        Uniform random input sequence.
    y : np.ndarray of shape (n_steps,)
        Target sequence y(k + 1) aligned with input u(k).
    """
    if n_steps <= 10:
        raise ValueError("n_steps must be > 10.")
    gen = _get_rng(seed)
    low, high = float(u_range[0]), float(u_range[1])
    total = n_steps + 1
    u = gen.uniform(low, high, size=total)
    y = np.zeros(total, dtype=np.float64)
    for k in range(9, total - 1):
        y[k + 1] = (
            0.3 * y[k]
            + 0.05 * y[k] * float(np.sum(y[k - 9 : k + 1]))
            + 1.5 * u[k - 9] * u[k]
            + 0.1
        )
    # Input u[k] predicts y[k + 1].
    return u[:-1, None], y[1:]


def generate_memory_capacity_data(
    n_steps: int = 2000,
    max_delay: int = 100,
    u_range: Tuple[float, float] = (-0.8, 0.8),
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Jaeger's memory capacity benchmark (Section 2 & Appendix A.2).

    Generates uniform random input u(t) in [-0.8, 0.8] and multi-output targets
    Y[t, k - 1] = u(t - k) for delays k = 1, ..., max_delay.

    Returns
    -------
    u : np.ndarray of shape (n_steps, 1)
    targets : np.ndarray of shape (n_steps, max_delay)
    """
    if n_steps <= max_delay:
        raise ValueError("n_steps must be greater than max_delay.")
    if max_delay < 1:
        raise ValueError("max_delay must be >= 1.")
    gen = _get_rng(seed)
    low, high = float(u_range[0]), float(u_range[1])
    raw = gen.uniform(low, high, size=n_steps + max_delay)
    u = raw[max_delay:, None]
    targets = np.zeros((n_steps, max_delay), dtype=np.float64)
    for delay in range(1, max_delay + 1):
        targets[:, delay - 1] = raw[max_delay - delay : max_delay - delay + n_steps]
    return u, targets


def generate_mackey_glass(
    n_steps: int = 2000,
    tau: int = 17,
    dt: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_exp: int = 10,
    washout: int = 500,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate the Mackey-Glass chaotic timeseries (Section 3.1; Steil, 2005).

    Integrates dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)
    using 4th-order Runge-Kutta with step sub-division.
    """
    if n_steps < 1 or tau < 1 or dt <= 0.0:
        raise ValueError("n_steps >= 1, tau >= 1, and dt > 0 are required.")
    gen = _get_rng(seed)
    substeps = 10
    h = float(dt) / substeps
    delay_steps = int(round(tau / h))
    total_samples = n_steps + washout
    total_sub = total_samples * substeps

    hist = np.zeros(delay_steps + total_sub + 1, dtype=np.float64)
    hist[: delay_steps + 1] = 1.2 + 0.05 * gen.standard_normal(delay_steps + 1)

    def rhs(x_now: float, x_del: float) -> float:
        return beta * x_del / (1.0 + x_del ** n_exp) - gamma * x_now

    for idx in range(delay_steps, delay_steps + total_sub):
        x = hist[idx]
        x_del = hist[idx - delay_steps]
        k1 = rhs(x, x_del)
        k2 = rhs(x + 0.5 * h * k1, x_del)
        k3 = rhs(x + 0.5 * h * k2, x_del)
        k4 = rhs(x + h * k3, x_del)
        hist[idx + 1] = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    series = hist[delay_steps + washout * substeps :: substeps][:n_steps]
    return series


def generate_synthetic_cochleagram_digits(
    n_classes: int = 10,
    n_samples_per_class: int = 20,
    n_channels: int = 24,
    n_timesteps: int = 80,
    noise_std: float = 0.12,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate synthetic multi-channel spectro-temporal patterns (cochleagram-like).

    The paper's isolated spoken digit task (Appendix A.3) uses the proprietary
    TI46 corpus preprocessed with Lyon's Passive Ear model (77 channels). This
    generator constructs public, self-contained synthetic multi-channel
    trajectories with distinct time-varying formant sweeps per class, random
    temporal warping/shifts, speaker-like frequency shifts, and additive noise,
    enabling end-to-end testing of the classification and spiking BSA pipeline.

    Returns
    -------
    sequences : list of np.ndarray, each of shape (n_timesteps, n_channels)
        Non-negative cochleagram-like features in [0, 1].
    labels : np.ndarray of shape (n_classes * n_samples_per_class,)
        Integer class labels in [0, n_classes - 1].
    """
    if n_classes < 2 or n_samples_per_class < 1:
        raise ValueError("n_classes >= 2 and n_samples_per_class >= 1 required.")
    if n_channels < 4 or n_timesteps < 10:
        raise ValueError("n_channels >= 4 and n_timesteps >= 10 required.")

    gen = _get_rng(seed)
    t_grid = np.linspace(0.0, 1.0, n_timesteps)
    ch_grid = np.linspace(0.0, 1.0, n_channels)

    # Create class-specific spectro-temporal formant trajectories (two formants each).
    prototypes = []
    for cls in range(n_classes):
        f1_start = 0.15 + 0.65 * (cls / max(n_classes - 1, 1))
        f1_end = 0.15 + 0.65 * (((cls * 3 + 2) % n_classes) / max(n_classes - 1, 1))
        f2_start = 0.2 + 0.65 * (((cls * 5 + 1) % n_classes) / max(n_classes - 1, 1))
        f2_end = 0.2 + 0.65 * (((cls * 7 + 4) % n_classes) / max(n_classes - 1, 1))
        phase = 0.25 + 0.5 * ((cls % 4) / 3.0)
        prototypes.append((f1_start, f1_end, f2_start, f2_end, phase))

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    bandwidth = 0.08

    for cls in range(n_classes):
        f1_s, f1_e, f2_s, f2_e, phase = prototypes[cls]
        for _ in range(n_samples_per_class):
            # Speaker-like pitch shift and mild temporal stretch/shift.
            freq_shift = float(gen.uniform(-0.04, 0.04))
            time_shift = float(gen.uniform(-0.06, 0.06))
            t_warped = np.clip(t_grid + time_shift, 0.0, 1.0)

            f1 = f1_s + (f1_e - f1_s) * t_warped + freq_shift
            f2 = f2_s + (f2_e - f2_s) * (t_warped ** 1.5) - 0.5 * freq_shift

            env1 = np.exp(-0.5 * ((t_warped - phase) / 0.28) ** 2)
            env2 = np.exp(-0.5 * ((t_warped - (1.0 - phase)) / 0.28) ** 2)

            spec1 = env1[:, None] * np.exp(
                -0.5 * ((ch_grid[None, :] - f1[:, None]) / bandwidth) ** 2
            )
            spec2 = env2[:, None] * np.exp(
                -0.5 * ((ch_grid[None, :] - f2[:, None]) / bandwidth) ** 2
            )
            sample = 0.65 * spec1 + 0.55 * spec2
            sample += gen.normal(0.0, noise_std, size=sample.shape)
            sample = np.clip(sample, 0.0, 1.0)
            sequences.append(sample)
            labels.append(cls)

    order = gen.permutation(len(sequences))
    ordered_seqs = [sequences[i] for i in order]
    ordered_labels = np.array([labels[i] for i in order], dtype=np.int64)
    return ordered_seqs, ordered_labels
