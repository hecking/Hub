"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Optional, Tuple, Union

import numpy as np


def _to_2d(signal: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("signal must be a 1D (T,) or 2D (T, C) array.")


def exponential_filter_kernel(
    tau: float = 30.0,
    length: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Create a causal exponentially decaying filter kernel h[k] = exp(-k / tau).

    Default tau=30.0 matches Verstraeten et al. (2007, Appendix A.3).
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    k_len = int(round(3.0 * tau)) if length is None else int(length)
    k_len = max(k_len, 1)
    kernel = np.exp(-np.arange(k_len, dtype=np.float64) / float(tau))
    if normalize:
        total = float(np.sum(kernel))
        if total > 0.0:
            kernel /= total
    return kernel


def poisson_encode(
    signal: np.ndarray,
    max_rate: float = 1.0,
    dt: float = 1.0,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Encode an analog signal into spike trains via a Poisson process (Section 3.1).

    Instantaneous firing probability per timestep is clip(signal * max_rate * dt, 0, 1).
    """
    sig_2d, was_1d = _to_2d(signal)
    gen = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    probs = np.clip(sig_2d * float(max_rate) * float(dt), 0.0, 1.0)
    spikes = (gen.random(sig_2d.shape) < probs).astype(np.float64)
    return spikes[:, 0] if was_1d else spikes


def lif_encode(
    signal: np.ndarray,
    tau_m: float = 20.0,
    threshold: float = 1.0,
    reset: float = 0.0,
    dt: float = 1.0,
) -> np.ndarray:
    """Encode an analog signal into spike trains using LIF neurons (Section 3.1).

    Feeds each channel of the analog timeseries as membrane current into a
    Leaky Integrate-and-Fire neuron.
    """
    if tau_m <= 0.0 or dt <= 0.0:
        raise ValueError("tau_m and dt must be positive.")
    sig_2d, was_1d = _to_2d(signal)
    n_steps, n_channels = sig_2d.shape
    decay = float(np.exp(-dt / tau_m))
    v = np.full(n_channels, float(reset), dtype=np.float64)
    spikes = np.zeros((n_steps, n_channels), dtype=np.float64)

    for t in range(n_steps):
        v = decay * v + sig_2d[t] * dt
        fired = v >= threshold
        if np.any(fired):
            spikes[t, fired] = 1.0
            v[fired] = reset

    return spikes[:, 0] if was_1d else spikes


def bsa_encode(
    signal: np.ndarray,
    filter_kernel: Optional[np.ndarray] = None,
    tau: float = 30.0,
    filter_len: Optional[int] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Encode an analog timeseries using Ben's Spiker Algorithm (BSA).

    Implements Schrauwen & Van Campenhout (2003) as used in Verstraeten et al.
    (2007, Section 3.1 & Appendix A.3). At each timestep t, emits a spike and
    subtracts the filter kernel from the residual signal if doing so reduces
    L1 error by at least `threshold`.
    """
    sig_2d, was_1d = _to_2d(signal)
    kernel = (
        exponential_filter_kernel(tau=tau, length=filter_len, normalize=True)
        if filter_kernel is None
        else np.asarray(filter_kernel, dtype=np.float64).ravel()
    )
    if kernel.size == 0:
        raise ValueError("filter_kernel must be non-empty.")

    n_steps, n_channels = sig_2d.shape
    k_len = kernel.size
    residual = sig_2d.copy()
    spikes = np.zeros((n_steps, n_channels), dtype=np.float64)

    for t in range(n_steps):
        end = min(t + k_len, n_steps)
        m = end - t
        h = kernel[:m, None]
        window = residual[t:end]
        err_spike = np.sum(np.abs(window - h), axis=0)
        err_no_spike = np.sum(np.abs(window), axis=0)
        fired = err_spike <= (err_no_spike - threshold)
        if np.any(fired):
            spikes[t, fired] = 1.0
            residual[t:end, fired] -= h

    return spikes[:, 0] if was_1d else spikes


def bsa_decode(
    spikes: np.ndarray,
    filter_kernel: Optional[np.ndarray] = None,
    tau: float = 30.0,
    filter_len: Optional[int] = None,
) -> np.ndarray:
    """Decode BSA spike trains back to analog signals by causal filter convolution."""
    spk_2d, was_1d = _to_2d(spikes)
    kernel = (
        exponential_filter_kernel(tau=tau, length=filter_len, normalize=True)
        if filter_kernel is None
        else np.asarray(filter_kernel, dtype=np.float64).ravel()
    )
    n_steps, n_channels = spk_2d.shape
    out = np.zeros((n_steps, n_channels), dtype=np.float64)
    for ch in range(n_channels):
        out[:, ch] = np.convolve(spk_2d[:, ch], kernel, mode="full")[:n_steps]
    return out[:, 0] if was_1d else out


def exponential_lowpass_filter(
    spikes: np.ndarray,
    tau: float = 30.0,
    dt: float = 1.0,
    subsample: int = 1,
) -> np.ndarray:
    """Low-pass filter spiking reservoir outputs with an exponential membrane kernel.

    Implements the spiking-to-analog conversion and optional subsampling
    described in Verstraeten et al. (2007, Section 3.4 & Appendix A.3).
    """
    if tau <= 0.0 or dt <= 0.0:
        raise ValueError("tau and dt must be positive.")
    if subsample < 1:
        raise ValueError("subsample must be >= 1.")

    spk_2d, was_1d = _to_2d(spikes)
    n_steps, n_channels = spk_2d.shape
    decay = float(np.exp(-dt / tau))
    filtered = np.zeros((n_steps, n_channels), dtype=np.float64)
    state = np.zeros(n_channels, dtype=np.float64)

    for t in range(n_steps):
        state = decay * state + spk_2d[t]
        filtered[t] = state

    if subsample > 1:
        filtered = filtered[::subsample]
    return filtered[:, 0] if was_1d else filtered
