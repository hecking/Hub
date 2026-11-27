"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from hub.reservoir.benchmarks import (
    generate_memory_capacity_data,
    generate_narma10,
)
from hub.reservoir.encoding import bsa_encode, lif_encode, poisson_encode
from hub.reservoir.metrics import (
    memory_capacity,
    mse,
    nrmse,
    pseudo_lyapunov_exponent,
    word_error_rate,
)
from hub.reservoir.nodes import BaseReservoir, create_reservoir
from hub.reservoir.readout import (
    BPDCNetwork,
    LinearReadout,
    PostProcessingPipeline,
    speech_classification_pipeline,
)


def _encode_signal(
    signal: np.ndarray,
    encoding: Optional[str],
    encoding_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if encoding is None or encoding.lower() == "none":
        return np.asarray(signal, dtype=np.float64)
    kwargs = dict(encoding_kwargs or {})
    mode = encoding.lower()
    if mode == "bsa":
        return bsa_encode(signal, **kwargs)
    if mode == "poisson":
        return poisson_encode(signal, **kwargs)
    if mode == "lif":
        return lif_encode(signal, **kwargs)
    raise ValueError(
        f"Unsupported spike encoding '{encoding}'. "
        "Expected None, 'bsa', 'poisson', or 'lif'."
    )


class ReservoirPipeline:
    """End-to-end simulation pipeline mirroring the RC Toolbox (Section 3, Fig. 10).

    Connects:
      [Optional Spike Encoding] -> [Reservoir Simulation] ->
      [Linear Readout] -> [Optional Post-Processing Pipeline].
    """

    def __init__(
        self,
        reservoir: BaseReservoir,
        readout: Optional[LinearReadout] = None,
        post_processor: Optional[PostProcessingPipeline] = None,
        spike_encoding: Optional[str] = None,
        encoding_kwargs: Optional[Dict[str, Any]] = None,
        washout: int = 100,
    ):
        self.reservoir = reservoir
        self.readout = LinearReadout() if readout is None else readout
        self.post_processor = post_processor
        self.spike_encoding = spike_encoding
        self.encoding_kwargs = dict(encoding_kwargs or {})
        self.washout = int(washout)

    def collect_states(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        return_pre_activations: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        encoded = _encode_signal(
            inputs, self.spike_encoding, self.encoding_kwargs
        )
        return self.reservoir.run(
            encoded,
            reset=reset,
            return_pre_activations=return_pre_activations,
        )

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Run reservoir on training inputs, fit readout after washout."""
        states = self.collect_states(inputs, reset=reset)
        y = np.asarray(targets, dtype=np.float64)
        w = self.washout
        if states.shape[0] != y.shape[0]:
            # Align targets when spiking reservoir subsampled states.
            idx = np.linspace(0, y.shape[0] - 1, states.shape[0]).astype(int)
            y = y[idx]
        if w >= states.shape[0]:
            raise ValueError(
                f"washout ({w}) must be smaller than state length ({states.shape[0]})."
            )
        self.readout.fit(states[w:], y[w:])
        return states

    def predict(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        apply_post_processing: bool = True,
    ) -> Union[np.ndarray, int]:
        """Run reservoir and readout (plus optional post-processing) on inputs."""
        states = self.collect_states(inputs, reset=reset)
        raw_out = self.readout.predict(states)
        if apply_post_processing and self.post_processor is not None:
            return self.post_processor(raw_out)
        return raw_out


def evaluate_narma10(
    node_type: str = "tanh",
    n_neurons: int = 100,
    scale_metric: str = "spectral_radius",
    scale_value: float = 0.9,
    retainment_rate: float = 0.2,
    train_steps: int = 1000,
    test_steps: int = 1000,
    washout: int = 100,
    regularization: float = 0.0,
    compute_lyapunov: bool = True,
    lyapunov_sample_step: int = 10,
    seed: Optional[int] = None,
    **extra_kwargs,
) -> Dict[str, float]:
    """Evaluate a reservoir model on the NARMA-10 benchmark (Appendix A.1).

    Supports 'linear', 'tanh', 'integrator', and 'bpdc'.
    """
    rng = np.random.default_rng(seed)
    u_train, y_train = generate_narma10(
        n_steps=train_steps + washout, seed=int(rng.integers(0, 2 ** 31 - 1))
    )
    u_test, y_test = generate_narma10(
        n_steps=test_steps + washout, seed=int(rng.integers(0, 2 ** 31 - 1))
    )

    kind = node_type.lower()
    if kind == "bpdc":
        bpdc_kwargs = {
            "learning_rate": extra_kwargs.pop("learning_rate", 0.05),
            "leak_rate": extra_kwargs.pop("leak_rate", 0.8),
            "feedback_scale": extra_kwargs.pop("feedback_scale", 0.1),
            "epsilon": extra_kwargs.pop("epsilon", 1e-2),
        }
        epochs = int(extra_kwargs.pop("epochs", 5))
        net = BPDCNetwork(
            n_neurons=n_neurons,
            n_inputs=1,
            n_outputs=1,
            scale_metric=scale_metric,
            scale_value=scale_value,
            rng=rng,
            **bpdc_kwargs,
        )
        net.fit(u_train, y_train, washout=washout, epochs=epochs)
        y_pred = net.predict(u_test, reset=True)
        result = {
            "mse": mse(y_pred[washout:], y_test[washout:]),
            "nrmse": nrmse(y_pred[washout:], y_test[washout:]),
        }
        return result

    node_kwargs = dict(extra_kwargs)
    if kind in ("integrator", "analog_integrator", "leaky"):
        node_kwargs["retainment_rate"] = retainment_rate

    reservoir = create_reservoir(
        node_type=kind,
        n_neurons=n_neurons,
        n_inputs=1,
        scale_metric=scale_metric,
        scale_value=scale_value,
        rng=rng,
        **node_kwargs,
    )
    pipeline = ReservoirPipeline(
        reservoir=reservoir,
        readout=LinearReadout(regularization=regularization),
        washout=washout,
    )
    pipeline.fit(u_train, y_train)
    states_test, pre_test = pipeline.collect_states(
        u_test, reset=True, return_pre_activations=True
    )
    y_pred = pipeline.readout.predict(states_test)

    result = {
        "mse": mse(y_pred[washout:], y_test[washout:]),
        "nrmse": nrmse(y_pred[washout:], y_test[washout:]),
    }
    analog_types = ("linear", "tanh", "integrator", "analog_integrator", "leaky")
    if compute_lyapunov and kind in analog_types:
        is_integ = kind in ("integrator", "analog_integrator", "leaky")
        r_val = retainment_rate if is_integ else 0.0
        result["pseudo_lyapunov"] = float(
            pseudo_lyapunov_exponent(
                reservoir.weights,
                states_test,
                node_type=reservoir.node_type,
                retainment_rate=r_val,
                sample_step=lyapunov_sample_step,
                washout=washout,
                pre_activations=pre_test,
            )
        )
    return result


def evaluate_memory_capacity(
    node_type: str = "tanh",
    n_neurons: int = 100,
    max_delay: int = 100,
    scale_metric: str = "spectral_radius",
    scale_value: float = 0.9,
    retainment_rate: float = 0.2,
    train_steps: int = 1000,
    test_steps: int = 1000,
    washout: int = 200,
    regularization: float = 0.0,
    compute_lyapunov: bool = True,
    lyapunov_sample_step: int = 10,
    seed: Optional[int] = None,
    **extra_kwargs,
) -> Dict[str, Any]:
    """Evaluate a reservoir on Jaeger's memory capacity task (Appendix A.2)."""
    rng = np.random.default_rng(seed)
    u_train, y_train = generate_memory_capacity_data(
        n_steps=train_steps + washout,
        max_delay=max_delay,
        seed=int(rng.integers(0, 2 ** 31 - 1)),
    )
    u_test, y_test = generate_memory_capacity_data(
        n_steps=test_steps + washout,
        max_delay=max_delay,
        seed=int(rng.integers(0, 2 ** 31 - 1)),
    )

    kind = node_type.lower()
    node_kwargs = dict(extra_kwargs)
    if kind in ("integrator", "analog_integrator", "leaky"):
        node_kwargs["retainment_rate"] = retainment_rate

    reservoir = create_reservoir(
        node_type=kind,
        n_neurons=n_neurons,
        n_inputs=1,
        scale_metric=scale_metric,
        scale_value=scale_value,
        rng=rng,
        **node_kwargs,
    )
    pipeline = ReservoirPipeline(
        reservoir=reservoir,
        readout=LinearReadout(regularization=regularization),
        washout=washout,
    )
    pipeline.fit(u_train, y_train)
    states_test, pre_test = pipeline.collect_states(
        u_test, reset=True, return_pre_activations=True
    )
    y_pred = pipeline.readout.predict(states_test)
    mc_total, mc_delays = memory_capacity(
        y_pred[washout:], y_test[washout:], return_per_delay=True
    )

    result: Dict[str, Any] = {
        "mc": mc_total,
        "mc_per_delay": mc_delays,
    }
    analog_types = ("linear", "tanh", "integrator", "analog_integrator", "leaky")
    if compute_lyapunov and kind in analog_types:
        is_integ = kind in ("integrator", "analog_integrator", "leaky")
        r_val = retainment_rate if is_integ else 0.0
        result["pseudo_lyapunov"] = float(
            pseudo_lyapunov_exponent(
                reservoir.weights,
                states_test,
                node_type=reservoir.node_type,
                retainment_rate=r_val,
                sample_step=lyapunov_sample_step,
                washout=washout,
                pre_activations=pre_test,
            )
        )
    return result


def evaluate_classification_cv(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    node_type: str = "tanh",
    n_neurons: int = 50,
    n_folds: int = 5,
    scale_metric: str = "spectral_radius",
    scale_value: float = 0.9,
    retainment_rate: float = 0.2,
    spike_encoding: Optional[str] = None,
    encoding_kwargs: Optional[Dict[str, Any]] = None,
    regularization: float = 1e-4,
    seed: Optional[int] = None,
    **node_kwargs,
) -> Dict[str, float]:
    """Evaluate temporal sequence classification via k-fold CV (Appendix A.3).

    Trains one linear classifier per class to output +1 for the sample's class
    and -1 for other classes at every timestep, then classifies each test
    sequence via temporal mean + winner-take-all (`speech_classification_pipeline`),
    returning Word Error Rate (WER) and accuracy.
    """
    n_samples = len(sequences)
    y_labels = np.asarray(labels, dtype=np.int64).ravel()
    if n_samples != y_labels.size:
        raise ValueError("sequences and labels must have the same length.")
    if n_folds < 2 or n_folds > n_samples:
        raise ValueError("n_folds must be in [2, len(sequences)].")

    rng = np.random.default_rng(seed)
    n_inputs = int(np.asarray(sequences[0]).shape[-1])
    classes = np.unique(y_labels)
    n_classes = classes.size
    class_to_idx = {int(c): idx for idx, c in enumerate(classes)}

    kind = node_type.lower()
    kwargs = dict(node_kwargs)
    if kind in ("integrator", "analog_integrator", "leaky"):
        kwargs["retainment_rate"] = retainment_rate

    reservoir = create_reservoir(
        node_type=kind,
        n_neurons=n_neurons,
        n_inputs=n_inputs,
        scale_metric=scale_metric,
        scale_value=scale_value,
        rng=rng,
        **kwargs,
    )
    post = speech_classification_pipeline()

    # Precompute reservoir states for every sequence once per reservoir realization.
    all_states: List[np.ndarray] = []
    for seq in sequences:
        enc = _encode_signal(seq, spike_encoding, encoding_kwargs)
        st = reservoir.run(enc, reset=True)
        all_states.append(st)

    # Stratified-like fold assignment.
    fold_ids = np.zeros(n_samples, dtype=np.int64)
    for cls in classes:
        cls_indices = np.nonzero(y_labels == cls)[0]
        for pos, sample_idx in enumerate(cls_indices):
            fold_ids[sample_idx] = pos % n_folds

    pred_labels = np.zeros(n_samples, dtype=np.int64)
    for fold in range(n_folds):
        train_idx = np.nonzero(fold_ids != fold)[0]
        test_idx = np.nonzero(fold_ids == fold)[0]

        x_train = np.vstack([all_states[i] for i in train_idx])
        y_blocks = []
        for i in train_idx:
            t_len = all_states[i].shape[0]
            block = np.full((t_len, n_classes), -1.0, dtype=np.float64)
            block[:, class_to_idx[int(y_labels[i])]] = 1.0
            y_blocks.append(block)
        y_train = np.vstack(y_blocks)

        readout = LinearReadout(regularization=regularization)
        readout.fit(x_train, y_train)

        for i in test_idx:
            raw = readout.predict(all_states[i])
            pred_cls_idx = int(post(raw))
            pred_labels[i] = int(classes[pred_cls_idx])

    wer = word_error_rate(pred_labels, y_labels)
    return {"wer": wer, "accuracy": 1.0 - wer}


def parameter_sweep(
    param_grid: Dict[str, Sequence[Any]],
    run_fn: Callable[..., Dict[str, Any]],
    n_runs: int = 5,
    base_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run a Cartesian product parameter sweep over stochastic runs (Section 3).

    For each combination in `param_grid`, executes `run_fn(seed=..., **params)`
    `n_runs` times and records mean and standard deviation for every scalar metric.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    results: List[Dict[str, Any]] = []

    for combo_idx, values in enumerate(itertools.product(*value_lists)):
        params = dict(zip(keys, values))
        run_outputs = []
        for run_idx in range(n_runs):
            seed = base_seed + combo_idx * 1000 + run_idx
            run_outputs.append(run_fn(seed=seed, **params))

        entry: Dict[str, Any] = {"params": params, "n_runs": n_runs}
        scalar_keys = [
            k
            for k, v in run_outputs[0].items()
            if isinstance(v, (int, float, np.floating, np.integer))
        ]
        for sk in scalar_keys:
            vals = np.array([float(r[sk]) for r in run_outputs], dtype=np.float64)
            entry[f"{sk}_mean"] = float(np.mean(vals))
            entry[f"{sk}_std"] = float(np.std(vals))
        results.append(entry)

    return results
