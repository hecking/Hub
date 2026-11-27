"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np

from hub.reservoir.encoding import exponential_lowpass_filter
from hub.reservoir.topology import generate_input_weights, generate_reservoir_weights


class LinearReadout:
    """Linear readout layer trained by Moore-Penrose pseudo-inverse or ridge regression.

    Matches Verstraeten et al. (2007, Section 3.4 & Appendix A.2), where the
    default (regularization=0.0) computes the exact Moore-Penrose pseudo-inverse.
    """

    def __init__(
        self,
        regularization: float = 0.0,
        fit_intercept: bool = True,
    ):
        if regularization < 0.0:
            raise ValueError("regularization must be >= 0.")
        self.regularization = float(regularization)
        self.fit_intercept = bool(fit_intercept)
        self.weights_: Optional[np.ndarray] = None
        self._target_was_1d: bool = False

    def _augment(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if not self.fit_intercept:
            return x
        return np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])

    def fit(self, states: np.ndarray, targets: np.ndarray) -> "LinearReadout":
        """Fit linear readout weights mapping reservoir states (T, N) to targets."""
        x = self._augment(states)
        y = np.asarray(targets, dtype=np.float64)
        self._target_was_1d = y.ndim == 1
        if y.ndim == 1:
            y = y[:, None]
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"states has {x.shape[0]} samples but targets has {y.shape[0]}."
            )

        if self.regularization == 0.0:
            self.weights_ = np.linalg.pinv(x) @ y
        else:
            n_feat = x.shape[1]
            reg_mat = self.regularization * np.eye(n_feat, dtype=np.float64)
            if self.fit_intercept:
                reg_mat[-1, -1] = 0.0
            self.weights_ = np.linalg.solve(x.T @ x + reg_mat, x.T @ y)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict outputs for reservoir states (T, N)."""
        if self.weights_ is None:
            raise RuntimeError("LinearReadout must be fitted before calling predict.")
        x = self._augment(states)
        out = x @ self.weights_
        if self._target_was_1d:
            return out[:, 0]
        return out


def _apply_nonlinearity(
    arr: np.ndarray,
    op: Union[str, Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    if callable(op):
        return np.asarray(op(arr))
    name = op.lower()
    if name == "identity":
        return arr
    if name == "tanh":
        return np.tanh(arr)
    if name == "sign":
        return np.sign(arr)
    if name in ("wta", "winner_take_all"):
        if arr.ndim == 1:
            return int(np.argmax(arr))
        return np.argmax(arr, axis=-1)
    raise ValueError(
        f"Unsupported nonlinearity '{op}'. Expected 'identity', 'tanh', "
        "'sign', 'wta', or a callable."
    )


def _apply_temporal_filter(
    arr: np.ndarray,
    op: Union[str, Callable[[np.ndarray], np.ndarray]],
    lowpass_tau: float = 10.0,
) -> np.ndarray:
    if callable(op):
        return np.asarray(op(arr))
    name = op.lower()
    if name == "identity":
        return arr
    if name == "mean":
        return np.mean(arr, axis=0)
    if name == "lowpass":
        return exponential_lowpass_filter(arr, tau=lowpass_tau, dt=1.0, subsample=1)
    raise ValueError(
        f"Unsupported temporal_filter '{op}'. Expected 'identity', 'mean', "
        "'lowpass', or a callable."
    )


class PostProcessingPipeline:
    """Modular 3-stage output post-processing pipeline (Section 3.4).

    Applies: nonlinearity1 -> temporal_filter -> nonlinearity2.
    For speech / digit classification (Appendix A.3), uses:
        identity -> mean -> wta (winner-take-all).
    """

    def __init__(
        self,
        nonlinearity1: Union[str, Callable[[np.ndarray], np.ndarray]] = "identity",
        temporal_filter: Union[str, Callable[[np.ndarray], np.ndarray]] = "identity",
        nonlinearity2: Union[str, Callable[[np.ndarray], np.ndarray]] = "identity",
        lowpass_tau: float = 10.0,
    ):
        self.nonlinearity1 = nonlinearity1
        self.temporal_filter = temporal_filter
        self.nonlinearity2 = nonlinearity2
        self.lowpass_tau = float(lowpass_tau)

    def __call__(self, outputs: np.ndarray) -> Union[np.ndarray, int]:
        x = np.asarray(outputs, dtype=np.float64)
        x = _apply_nonlinearity(x, self.nonlinearity1)
        x = _apply_temporal_filter(
            x, self.temporal_filter, lowpass_tau=self.lowpass_tau
        )
        return _apply_nonlinearity(x, self.nonlinearity2)


def speech_classification_pipeline() -> PostProcessingPipeline:
    """Return the temporal-mean + winner-take-all pipeline from Appendix A.3."""
    return PostProcessingPipeline(
        nonlinearity1="identity",
        temporal_filter="mean",
        nonlinearity2="wta",
    )


class BPDCNetwork:
    """Backpropagation-Decorrelation (BPDC) O(N) online recurrent network.

    Implements Steil (2004, 2005) as compared against ESNs and LSMs in
    Verstraeten et al. (2007, Sections 1.1, 2.1, 3.4):
      - Fixed internal reservoir weights W and input weights W_in.
      - Feedback connections from the output layer into the reservoir (W_fb)
        and recurrent connections inside the output layer (W_out_rec).
      - Pre-nonlinearity leaky state integration (noted in Section 2.1).
      - Online O(N) decorrelated readout weight update rule with teacher
        forcing during training.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        n_inputs: int = 1,
        n_outputs: int = 1,
        weights: Optional[np.ndarray] = None,
        input_weights: Optional[np.ndarray] = None,
        feedback_weights: Optional[np.ndarray] = None,
        scale_metric: Optional[str] = "spectral_radius",
        scale_value: float = 0.9,
        connection_fraction: float = 0.5,
        input_connection_fraction: float = 0.1,
        feedback_scale: float = 0.2,
        learning_rate: float = 0.05,
        leak_rate: float = 0.8,
        epsilon: float = 1e-2,
        output_activation: str = "linear",
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        if not 0.0 < leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1].")
        if output_activation not in ("linear", "tanh"):
            raise ValueError("output_activation must be 'linear' or 'tanh'.")

        gen = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
        )
        if weights is None:
            weights = generate_reservoir_weights(
                n_neurons=n_neurons,
                connection_fraction=connection_fraction,
                scale_metric=scale_metric,
                scale_value=scale_value,
                rng=gen,
            )
        if input_weights is None:
            input_weights = generate_input_weights(
                n_neurons=weights.shape[0],
                n_inputs=n_inputs,
                connection_fraction=input_connection_fraction,
                rng=gen,
            )

        self.weights = np.asarray(weights, dtype=np.float64)
        self.input_weights = np.asarray(input_weights, dtype=np.float64)
        if self.input_weights.ndim == 1:
            self.input_weights = self.input_weights[:, None]
        self.n_neurons = self.weights.shape[0]
        self.n_inputs = self.input_weights.shape[1]
        self.n_outputs = int(n_outputs)

        if feedback_weights is None:
            self.feedback_weights = (
                gen.uniform(-1.0, 1.0, size=(self.n_neurons, self.n_outputs))
                * float(feedback_scale)
            )
        else:
            self.feedback_weights = np.asarray(feedback_weights, dtype=np.float64)

        # Trainable weights: reservoir->output, output->output (recurrent), and bias.
        self.w_out = np.zeros((self.n_outputs, self.n_neurons), dtype=np.float64)
        self.w_out_rec = np.zeros(
            (self.n_outputs, self.n_outputs), dtype=np.float64
        )
        self.b_out = np.zeros(self.n_outputs, dtype=np.float64)

        self.learning_rate = float(learning_rate)
        self.leak_rate = float(leak_rate)
        self.epsilon = float(epsilon)
        self.output_activation = output_activation
        self.reset()

    def reset(self) -> None:
        self._res_potential = np.zeros(self.n_neurons, dtype=np.float64)
        self._res_state = np.zeros(self.n_neurons, dtype=np.float64)
        self._out_state = np.zeros(self.n_outputs, dtype=np.float64)
        self._prev_error = np.zeros(self.n_outputs, dtype=np.float64)

    def _activate_output(self, pre: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.output_activation == "tanh":
            act = np.tanh(pre)
            deriv = np.maximum(1.0 - act ** 2, 1e-3)
            return act, deriv
        return pre, np.ones_like(pre)

    def _step_reservoir(self, u_t: np.ndarray, fb_signal: np.ndarray) -> np.ndarray:
        alpha = self.leak_rate
        drive = (
            self.input_weights @ u_t
            + self.weights @ self._res_state
            + self.feedback_weights @ fb_signal
        )
        self._res_potential = (1.0 - alpha) * self._res_potential + alpha * drive
        self._res_state = np.tanh(self._res_potential)
        return self._res_state

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        washout: int = 50,
        epochs: int = 1,
    ) -> np.ndarray:
        """Train output and output-recurrent weights online using BPDC.

        Returns the training predictions from the final epoch.
        """
        u = np.asarray(inputs, dtype=np.float64)
        if u.ndim == 1:
            u = u[:, None]
        y = np.asarray(targets, dtype=np.float64)
        target_was_1d = y.ndim == 1
        if y.ndim == 1:
            y = y[:, None]

        n_steps = u.shape[0]
        preds = np.zeros((n_steps, self.n_outputs), dtype=np.float64)
        alpha = self.leak_rate
        decorr_coeff = (1.0 - alpha) / alpha

        for _ in range(max(int(epochs), 1)):
            self.reset()
            for t in range(n_steps):
                fb_in = y[t - 1] if t > 0 else np.zeros(self.n_outputs)
                s_t = self._step_reservoir(u[t], fb_in)
                pre_out = (
                    self.w_out @ s_t + self.w_out_rec @ fb_in + self.b_out
                )
                y_hat, f_prime = self._activate_output(pre_out)
                preds[t] = y_hat

                err = y[t] - y_hat
                if t >= washout:
                    delta = (err - decorr_coeff * self._prev_error) / f_prime
                    norm_sq = (
                        float(np.dot(s_t, s_t))
                        + float(np.dot(fb_in, fb_in))
                        + 1.0
                        + self.epsilon
                    )
                    step = (self.learning_rate / norm_sq) * delta
                    self.w_out += np.outer(step, s_t)
                    self.w_out_rec += np.outer(step, fb_in)
                    self.b_out += step
                self._prev_error = err
                self._out_state = y[t]

        return preds[:, 0] if target_was_1d else preds

    def predict(
        self,
        inputs: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Run closed-loop inference with output feedback."""
        u = np.asarray(inputs, dtype=np.float64)
        if u.ndim == 1:
            u = u[:, None]
        if reset:
            self.reset()

        n_steps = u.shape[0]
        preds = np.zeros((n_steps, self.n_outputs), dtype=np.float64)
        for t in range(n_steps):
            fb_in = self._out_state.copy()
            s_t = self._step_reservoir(u[t], fb_in)
            pre_out = self.w_out @ s_t + self.w_out_rec @ fb_in + self.b_out
            y_hat, _ = self._activate_output(pre_out)
            y_hat = np.clip(y_hat, -1e3, 1e3)
            self._out_state = y_hat
            preds[t] = y_hat
        return preds[:, 0] if self.n_outputs == 1 else preds
