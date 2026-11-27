"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import heapq
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from hub.reservoir.encoding import exponential_lowpass_filter
from hub.reservoir.topology import generate_input_weights, generate_reservoir_weights


def _ensure_input_2d(inputs: np.ndarray) -> np.ndarray:
    arr = np.asarray(inputs, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("inputs must be 1D (T,) or 2D (T, n_inputs).")


class BaseReservoir:
    """Base interface for all reservoir node implementations (Section 3.3)."""

    node_type: str = "base"

    def __init__(
        self,
        weights: np.ndarray,
        input_weights: np.ndarray,
    ):
        w = np.asarray(weights, dtype=np.float64)
        w_in = np.asarray(input_weights, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError("weights must be a square (N, N) array.")
        if w_in.ndim == 1:
            w_in = w_in[:, None]
        if w_in.ndim != 2 or w_in.shape[0] != w.shape[0]:
            raise ValueError(
                f"input_weights must have shape ({w.shape[0]}, n_inputs), "
                f"got {w_in.shape}."
            )
        self.weights = w
        self.input_weights = w_in
        self.n_neurons = w.shape[0]
        self.n_inputs = w_in.shape[1]
        self.state = np.zeros(self.n_neurons, dtype=np.float64)

    def reset(self) -> None:
        """Reset internal reservoir state to zero."""
        self.state = np.zeros(self.n_neurons, dtype=np.float64)

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate the reservoir on an input timeseries of shape (T, n_inputs)."""
        raise NotImplementedError


class LinearReservoir(BaseReservoir):
    """Reservoir composed of linear nodes: s(k+1) = W_in u(k) + W s(k)."""

    node_type = "linear"

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        u = _ensure_input_2d(inputs)
        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input channels, got {u.shape[1]}."
            )
        if reset:
            self.reset()

        n_steps = u.shape[0]
        states = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        s = self.state.copy()
        w = self.weights
        drive = u @ self.input_weights.T

        for t in range(n_steps):
            s = drive[t] + w @ s
            states[t] = s

        self.state = s
        if return_pre_activations:
            return states, states.copy()
        return states


class TanhReservoir(BaseReservoir):
    """Classic Echo State Network reservoir with tanh nodes (Section 2.1)."""

    node_type = "tanh"

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        u = _ensure_input_2d(inputs)
        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input channels, got {u.shape[1]}."
            )
        if reset:
            self.reset()

        n_steps = u.shape[0]
        states = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        pre_acts = (
            np.zeros((n_steps, self.n_neurons), dtype=np.float64)
            if return_pre_activations
            else None
        )
        s = self.state.copy()
        w = self.weights
        drive = u @ self.input_weights.T

        for t in range(n_steps):
            z = drive[t] + w @ s
            if pre_acts is not None:
                pre_acts[t] = z
            s = np.tanh(z)
            states[t] = s

        self.state = s
        if return_pre_activations:
            return states, pre_acts
        return states


class AnalogIntegratorReservoir(BaseReservoir):
    """Analog integrator (leaky) sigmoidal reservoir (Verstraeten et al., 2007, Eq. 1).

    Update equation with timestep delta and integration timeconstant tau_int:

        s(k + 1) = r * s(k) + (1 - r) * tanh(W_in u(k) + W s(k))

    where r = 1 - delta / tau_int is the retainment rate in [0, 1]. When r = 0,
    this reduces identically to TanhReservoir.
    """

    node_type = "integrator"

    def __init__(
        self,
        weights: np.ndarray,
        input_weights: np.ndarray,
        retainment_rate: Optional[float] = None,
        tau_int: Optional[float] = None,
        dt: float = 1.0,
    ):
        super().__init__(weights=weights, input_weights=input_weights)
        if retainment_rate is not None and tau_int is not None:
            raise ValueError("Specify either retainment_rate or tau_int, not both.")
        if tau_int is not None:
            if tau_int < dt or tau_int <= 0.0:
                raise ValueError("tau_int must be >= dt and > 0.")
            r = 1.0 - float(dt) / float(tau_int)
        elif retainment_rate is not None:
            r = float(retainment_rate)
        else:
            r = 0.2
        if not 0.0 <= r <= 1.0:
            raise ValueError("retainment_rate must be in [0, 1].")
        self.retainment_rate = r
        self.dt = float(dt)

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        u = _ensure_input_2d(inputs)
        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input channels, got {u.shape[1]}."
            )
        if reset:
            self.reset()

        n_steps = u.shape[0]
        states = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        pre_acts = (
            np.zeros((n_steps, self.n_neurons), dtype=np.float64)
            if return_pre_activations
            else None
        )
        s = self.state.copy()
        w = self.weights
        drive = u @ self.input_weights.T
        r = self.retainment_rate
        one_minus_r = 1.0 - r

        for t in range(n_steps):
            z = drive[t] + w @ s
            if pre_acts is not None:
                pre_acts[t] = z
            s = r * s + one_minus_r * np.tanh(z)
            states[t] = s

        self.state = s
        if return_pre_activations:
            return states, pre_acts
        return states


class LIFReservoir(BaseReservoir):
    """Leaky Integrate-and-Fire (LIF) spiking reservoir without synapse model.

    Implements the LIF neuron dynamics from Verstraeten et al. (2007, Section 2.1
    & Appendix B). Outputs either raw spike trains or exponentially low-pass
    filtered (and optionally subsampled) analog states.
    """

    node_type = "lif"

    def __init__(
        self,
        weights: np.ndarray,
        input_weights: np.ndarray,
        tau_m: float = 20.0,
        threshold: float = 1.0,
        reset_potential: float = 0.0,
        refractory_steps: int = 0,
        dt: float = 1.0,
        filter_tau: Optional[float] = 30.0,
        subsample: int = 1,
    ):
        super().__init__(weights=weights, input_weights=input_weights)
        if tau_m <= 0.0 or dt <= 0.0:
            raise ValueError("tau_m and dt must be positive.")
        if refractory_steps < 0:
            raise ValueError("refractory_steps must be >= 0.")
        self.tau_m = float(tau_m)
        self.threshold = float(threshold)
        self.reset_potential = float(reset_potential)
        self.refractory_steps = int(refractory_steps)
        self.dt = float(dt)
        self.filter_tau = filter_tau
        self.subsample = int(subsample)
        self.refractory_counter = np.zeros(self.n_neurons, dtype=np.int64)
        self.last_spikes = np.zeros(self.n_neurons, dtype=np.float64)

    def reset(self) -> None:
        super().reset()
        self.state = np.full(
            self.n_neurons, self.reset_potential, dtype=np.float64
        )
        self.refractory_counter = np.zeros(self.n_neurons, dtype=np.int64)
        self.last_spikes = np.zeros(self.n_neurons, dtype=np.float64)

    def run_spikes(
        self,
        inputs: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Simulate and return raw binary spike trains of shape (T, N)."""
        u = _ensure_input_2d(inputs)
        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input channels, got {u.shape[1]}."
            )
        if reset:
            self.reset()

        n_steps = u.shape[0]
        spikes = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        decay_m = float(np.exp(-self.dt / self.tau_m))
        v = self.state.copy()
        prev_spk = self.last_spikes.copy()
        refrac = self.refractory_counter.copy()
        w = self.weights
        in_drive = u @ self.input_weights.T

        for t in range(n_steps):
            active = refrac == 0
            v = decay_m * v + in_drive[t] + w @ prev_spk
            v[~active] = self.reset_potential
            fired = active & (v >= self.threshold)
            cur_spk = fired.astype(np.float64)
            if np.any(fired):
                v[fired] = self.reset_potential
                if self.refractory_steps > 0:
                    refrac[fired] = self.refractory_steps + 1
            refrac = np.maximum(refrac - 1, 0)
            spikes[t] = cur_spk
            prev_spk = cur_spk

        self.state = v
        self.last_spikes = prev_spk
        self.refractory_counter = refrac
        return spikes

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> np.ndarray:
        spikes = self.run_spikes(inputs, reset=reset)
        if self.filter_tau is None:
            return spikes[:: self.subsample] if self.subsample > 1 else spikes
        return exponential_lowpass_filter(
            spikes,
            tau=self.filter_tau,
            dt=self.dt,
            subsample=self.subsample,
        )


class BooijLIFReservoir(BaseReservoir):
    """LIF reservoir with the Booij simplified exponential synapse (Appendix B).

    Uses synaptic timeconstant tau_s = tau_m / 2 with PSP kernel
    exp(-t / tau_m) - exp(-t / tau_s). To match the PSP time-integral of LIF
    without a synapse (tau_m vs tau_m - tau_s = tau_m / 2; Eqs. B.2-B.3), the
    weights are automatically rescaled by 2.0 when `compensate_psp=True`.
    """

    node_type = "booij"

    def __init__(
        self,
        weights: np.ndarray,
        input_weights: np.ndarray,
        tau_m: float = 20.0,
        threshold: float = 1.0,
        reset_potential: float = 0.0,
        refractory_steps: int = 0,
        dt: float = 1.0,
        compensate_psp: bool = True,
        filter_tau: Optional[float] = 30.0,
        subsample: int = 1,
    ):
        super().__init__(weights=weights, input_weights=input_weights)
        if tau_m <= 0.0 or dt <= 0.0:
            raise ValueError("tau_m and dt must be positive.")
        if refractory_steps < 0:
            raise ValueError("refractory_steps must be >= 0.")
        self.tau_m = float(tau_m)
        self.tau_s = float(tau_m) / 2.0
        self.threshold = float(threshold)
        self.reset_potential = float(reset_potential)
        self.refractory_steps = int(refractory_steps)
        self.dt = float(dt)
        self.compensate_psp = bool(compensate_psp)
        self.psp_scale = 2.0 if compensate_psp else 1.0
        self.filter_tau = filter_tau
        self.subsample = int(subsample)
        self._a = np.zeros(self.n_neurons, dtype=np.float64)
        self._b = np.zeros(self.n_neurons, dtype=np.float64)
        self.refractory_counter = np.zeros(self.n_neurons, dtype=np.int64)
        self.last_spikes = np.zeros(self.n_neurons, dtype=np.float64)

    def reset(self) -> None:
        super().reset()
        self._a = np.zeros(self.n_neurons, dtype=np.float64)
        self._b = np.zeros(self.n_neurons, dtype=np.float64)
        self.state = np.full(
            self.n_neurons, self.reset_potential, dtype=np.float64
        )
        self.refractory_counter = np.zeros(self.n_neurons, dtype=np.int64)
        self.last_spikes = np.zeros(self.n_neurons, dtype=np.float64)

    def run_spikes(
        self,
        inputs: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Simulate and return raw binary spike trains of shape (T, N)."""
        u = _ensure_input_2d(inputs)
        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input channels, got {u.shape[1]}."
            )
        if reset:
            self.reset()

        n_steps = u.shape[0]
        spikes = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        decay_m = float(np.exp(-self.dt / self.tau_m))
        decay_s = float(np.exp(-self.dt / self.tau_s))

        a = self._a.copy()
        b = self._b.copy()
        prev_spk = self.last_spikes.copy()
        refrac = self.refractory_counter.copy()
        w_eff = self.weights * self.psp_scale
        in_drive = (u @ self.input_weights.T) * self.psp_scale

        for t in range(n_steps):
            drive = in_drive[t] + w_eff @ prev_spk
            a = decay_m * a + drive
            b = decay_s * b + drive
            v = self.reset_potential + (a - b)
            active = refrac == 0
            fired = active & (v >= self.threshold)
            cur_spk = fired.astype(np.float64)
            if np.any(fired):
                a[fired] -= self.threshold - self.reset_potential
                if self.refractory_steps > 0:
                    refrac[fired] = self.refractory_steps + 1
            refrac = np.maximum(refrac - 1, 0)
            spikes[t] = cur_spk
            prev_spk = cur_spk

        self._a = a
        self._b = b
        self.state = self.reset_potential + (a - b)
        self.last_spikes = prev_spk
        self.refractory_counter = refrac
        return spikes

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> np.ndarray:
        spikes = self.run_spikes(inputs, reset=reset)
        if self.filter_tau is None:
            return spikes[:: self.subsample] if self.subsample > 1 else spikes
        return exponential_lowpass_filter(
            spikes,
            tau=self.filter_tau,
            dt=self.dt,
            subsample=self.subsample,
        )


def booij_next_spike_time(
    a: float,
    b: float,
    tau_m: float,
    threshold: float,
    reset_potential: float = 0.0,
) -> Optional[float]:
    """Analytically solve the Booij quadratic firing-time equation (Appendix B).

    For tau_s = tau_m / 2 and membrane potential
        u(dt) = reset_potential + a * exp(-dt / tau_m) - b * exp(-2 * dt / tau_m),
    setting x = exp(-dt / tau_m) in (0, 1] yields the quadratic equation:
        b * x^2 - a * x + (threshold - reset_potential) = 0.

    Returns the earliest future crossing time dt >= 0, or None if the neuron
    will not cross threshold without further incoming spikes.
    """
    if tau_m <= 0.0:
        raise ValueError("tau_m must be positive.")
    theta = float(threshold) - float(reset_potential)
    if theta <= 0.0:
        return 0.0
    if a - b >= theta:
        return 0.0
    if b <= 0.0:
        return None
    if a <= 0.0 or a >= 2.0 * b:
        return None

    disc = a * a - 4.0 * b * theta
    if disc < 0.0:
        return None
    x_plus = (a + float(np.sqrt(disc))) / (2.0 * b)
    if not (0.0 < x_plus <= 1.0):
        return None
    return float(-tau_m * np.log(x_plus))


class EventBasedBooijReservoir(BaseReservoir):
    """Exact event-based simulator for Booij LIF reservoirs (Appendix B).

    Uses a priority queue of spike events and the analytical quadratic
    firing-time solution `booij_next_spike_time` (tau_s = tau_m / 2) to avoid
    timestep discretization errors, mirroring ESSpiNN (Section 3.3, Appendix B).
    """

    node_type = "booij_event"

    def __init__(
        self,
        weights: np.ndarray,
        input_weights: np.ndarray,
        tau_m: float = 20.0,
        threshold: float = 1.0,
        reset_potential: float = 0.0,
        synaptic_delay: float = 1.0,
        compensate_psp: bool = True,
        dt: float = 1.0,
        filter_tau: Optional[float] = 30.0,
        subsample: int = 1,
        max_events: int = 200000,
    ):
        super().__init__(weights=weights, input_weights=input_weights)
        if tau_m <= 0.0 or synaptic_delay <= 0.0 or dt <= 0.0:
            raise ValueError("tau_m, synaptic_delay, and dt must be positive.")
        self.tau_m = float(tau_m)
        self.tau_s = float(tau_m) / 2.0
        self.threshold = float(threshold)
        self.reset_potential = float(reset_potential)
        self.synaptic_delay = float(synaptic_delay)
        self.psp_scale = 2.0 if compensate_psp else 1.0
        self.dt = float(dt)
        self.filter_tau = filter_tau
        self.subsample = int(subsample)
        self.max_events = int(max_events)

    def simulate_events(
        self,
        input_events: List[Tuple[float, int]],
        t_end: float,
    ) -> Dict[int, List[float]]:
        """Simulate continuous-time spike events up to t_end.

        Parameters
        ----------
        input_events : list of (time, channel_index)
            External spike arrival times.
        t_end : float
            Simulation end time.

        Returns
        -------
        dict mapping neuron_index -> list of emitted spike times.
        """
        n = self.n_neurons
        a = np.zeros(n, dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        last_t = np.zeros(n, dtype=np.float64)
        version = np.zeros(n, dtype=np.int64)
        w_eff = self.weights * self.psp_scale
        w_in_eff = self.input_weights * self.psp_scale

        # Precompute outgoing adjacency lists for sparse event propagation.
        out_targets = [np.nonzero(w_eff[:, j])[0] for j in range(n)]
        in_targets = [
            np.nonzero(w_in_eff[:, c])[0] for c in range(self.n_inputs)
        ]

        # Heap items: (time, kind, target_idx, payload, expected_version)
        # kind: 0 = input arrival, 1 = recurrent arrival, 2 = neuron firing
        heap: List[Tuple[float, int, int, float, int]] = []
        for t_ev, ch in input_events:
            if 0.0 <= t_ev < t_end:
                heapq.heappush(heap, (float(t_ev), 0, int(ch), 1.0, 0))

        output_spikes: Dict[int, List[float]] = {i: [] for i in range(n)}
        processed = 0

        def advance_and_reschedule(idx: int, now: float, delta_w: float) -> None:
            dt_i = now - last_t[idx]
            if dt_i > 0.0:
                em = float(np.exp(-dt_i / self.tau_m))
                a[idx] *= em
                b[idx] *= em * em
                last_t[idx] = now
            a[idx] += delta_w
            b[idx] += delta_w
            version[idx] += 1
            dt_fire = booij_next_spike_time(
                a[idx], b[idx], self.tau_m, self.threshold, self.reset_potential
            )
            if dt_fire is not None and now + dt_fire < t_end:
                heapq.heappush(
                    heap, (now + dt_fire, 2, idx, 0.0, int(version[idx]))
                )

        while heap and processed < self.max_events:
            now, kind, idx, payload, ver = heapq.heappop(heap)
            if now >= t_end:
                break
            processed += 1
            if kind == 0:
                for tgt in in_targets[idx]:
                    advance_and_reschedule(int(tgt), now, float(w_in_eff[tgt, idx]))
            elif kind == 1:
                advance_and_reschedule(idx, now, payload)
            else:
                if ver != version[idx]:
                    continue
                dt_i = now - last_t[idx]
                if dt_i > 0.0:
                    em = float(np.exp(-dt_i / self.tau_m))
                    a[idx] *= em
                    b[idx] *= em * em
                    last_t[idx] = now
                output_spikes[idx].append(now)
                a[idx] -= self.threshold - self.reset_potential
                version[idx] += 1
                dt_fire = booij_next_spike_time(
                    a[idx], b[idx], self.tau_m, self.threshold, self.reset_potential
                )
                if dt_fire is not None and dt_fire > 1e-9 and now + dt_fire < t_end:
                    heapq.heappush(
                        heap, (now + dt_fire, 2, idx, 0.0, int(version[idx]))
                    )
                t_arrive = now + self.synaptic_delay
                if t_arrive < t_end:
                    for tgt in out_targets[idx]:
                        heapq.heappush(
                            heap,
                            (t_arrive, 1, int(tgt), float(w_eff[tgt, idx]), 0),
                        )

        self.state = self.reset_potential + (a - b)
        return output_spikes

    def run_spikes(
        self,
        inputs: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Convert discrete spikes to continuous events, simulate, and rasterize."""
        u = _ensure_input_2d(inputs)
        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input channels, got {u.shape[1]}."
            )
        if reset:
            self.reset()

        n_steps = u.shape[0]
        t_indices, ch_indices = np.nonzero(u > 0.0)
        events = [
            (float(t) * self.dt, int(c)) for t, c in zip(t_indices, ch_indices)
        ]
        spike_dict = self.simulate_events(events, t_end=n_steps * self.dt)

        raster = np.zeros((n_steps, self.n_neurons), dtype=np.float64)
        for neuron_idx, times in spike_dict.items():
            for t_spk in times:
                bin_idx = int(t_spk // self.dt)
                if 0 <= bin_idx < n_steps:
                    raster[bin_idx, neuron_idx] += 1.0
        return raster

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> np.ndarray:
        spikes = self.run_spikes(inputs, reset=reset)
        if self.filter_tau is None:
            return spikes[:: self.subsample] if self.subsample > 1 else spikes
        return exponential_lowpass_filter(
            spikes,
            tau=self.filter_tau,
            dt=self.dt,
            subsample=self.subsample,
        )


def create_reservoir(
    node_type: str = "tanh",
    n_neurons: int = 100,
    n_inputs: int = 1,
    weights: Optional[np.ndarray] = None,
    input_weights: Optional[np.ndarray] = None,
    topology: str = "random",
    connection_fraction: float = 0.5,
    input_connection_fraction: float = 0.1,
    weight_distribution: str = "normal",
    scale_metric: Optional[str] = "spectral_radius",
    scale_value: float = 0.9,
    input_scale: float = 1.0,
    rng: Optional[Union[int, np.random.Generator]] = None,
    **node_kwargs,
) -> BaseReservoir:
    """Generate weights (if omitted) and instantiate a reservoir node model."""
    gen = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    if weights is None:
        weights = generate_reservoir_weights(
            n_neurons=n_neurons,
            topology=topology,
            connection_fraction=connection_fraction,
            weight_distribution=weight_distribution,
            scale_metric=scale_metric,
            scale_value=scale_value,
            rng=gen,
        )
    if input_weights is None:
        input_weights = generate_input_weights(
            n_neurons=weights.shape[0],
            n_inputs=n_inputs,
            connection_fraction=input_connection_fraction,
            weight_distribution=weight_distribution,
            input_scale=input_scale,
            rng=gen,
        )

    kind = node_type.lower()
    if kind == "linear":
        return LinearReservoir(weights=weights, input_weights=input_weights)
    if kind == "tanh":
        return TanhReservoir(weights=weights, input_weights=input_weights)
    if kind in ("integrator", "analog_integrator", "leaky"):
        return AnalogIntegratorReservoir(
            weights=weights, input_weights=input_weights, **node_kwargs
        )
    if kind == "lif":
        return LIFReservoir(
            weights=weights, input_weights=input_weights, **node_kwargs
        )
    if kind == "booij":
        return BooijLIFReservoir(
            weights=weights, input_weights=input_weights, **node_kwargs
        )
    if kind == "booij_event":
        return EventBasedBooijReservoir(
            weights=weights, input_weights=input_weights, **node_kwargs
        )
    raise ValueError(
        f"Unknown node_type '{node_type}'. Expected one of "
        "'linear', 'tanh', 'integrator', 'lif', 'booij', 'booij_event'."
    )
