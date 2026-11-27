"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

Runnable demo reproducing the core experimental findings of:
    D. Verstraeten, B. Schrauwen, M. D'Haene, D. Stroobandt (2007),
    "An experimental unification of reservoir computing methods",
    Neural Networks 20 (2007) 391-403.

Usage:
    python3 examples/reservoir_computing_demo.py
    python3 examples/reservoir_computing_demo.py --runs 5 --n-neurons 80
    python3 examples/reservoir_computing_demo.py --quick
"""

import argparse
import os
import sys

import numpy as np

# Allow running directly from the repository root without pip install -e .
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hub.reservoir as rc  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def experiment_node_complexity(n_runs: int, n_neurons: int) -> None:
    """Reproduce Figs. 2 & 3: node complexity vs spectral radius on MC & NARMA-10."""
    _print_header(
        "Experiment 1 (Figs. 2 & 3): Node Complexity vs. Spectral Radius"
    )
    spectral_radii = [0.5, 0.8, 1.0, 1.2, 1.5]

    print(f"\n[Memory Capacity] (N={n_neurons}, max_delay=60, runs={n_runs})")
    print(f"{'Node Type':<12} | " + " | ".join(f"rho={r:4.1f}" for r in spectral_radii))
    print("-" * 74)
    for node_type in ("linear", "tanh", "integrator"):
        row = []
        for rho in spectral_radii:
            res = rc.parameter_sweep(
                {"scale_value": [rho]},
                lambda seed, scale_value, nt=node_type: rc.evaluate_memory_capacity(
                    node_type=nt,
                    n_neurons=n_neurons,
                    max_delay=60,
                    scale_metric="spectral_radius",
                    scale_value=scale_value,
                    retainment_rate=0.2,
                    train_steps=800,
                    test_steps=800,
                    washout=150,
                    compute_lyapunov=False,
                    seed=seed,
                ),
                n_runs=n_runs,
                base_seed=10,
            )[0]
            row.append(f"{res['mc_mean']:8.2f}")
        print(f"{node_type:<12} | " + " | ".join(row))

    print(f"\n[NARMA-10 MSE] (N={n_neurons}, runs={n_runs})")
    print(f"{'Node Type':<12} | " + " | ".join(f"rho={r:4.1f}" for r in spectral_radii))
    print("-" * 74)
    for node_type in ("linear", "tanh", "integrator", "bpdc"):
        row = []
        for rho in spectral_radii:
            res = rc.parameter_sweep(
                {"scale_value": [rho]},
                lambda seed, scale_value, nt=node_type: rc.evaluate_narma10(
                    node_type=nt,
                    n_neurons=n_neurons,
                    scale_metric="spectral_radius",
                    scale_value=scale_value,
                    retainment_rate=0.2,
                    train_steps=800,
                    test_steps=800,
                    washout=100,
                    compute_lyapunov=False,
                    seed=seed,
                ),
                n_runs=n_runs,
                base_seed=100,
            )[0]
            mse_val = res["mse_mean"]
            cell = f"{mse_val:8.5f}" if mse_val < 10.0 else " >10.0  "
            row.append(cell)
        print(f"{node_type:<12} | " + " | ".join(row))


def experiment_node_vs_reservoir_memory(n_runs: int) -> None:
    """Reproduce Figs. 4 & 5: retainment rate r vs reservoir size."""
    _print_header(
        "Experiment 2 (Figs. 4 & 5): Node Memory (r) vs. Reservoir Size (N)"
    )
    sizes = [20, 50, 80]
    retainment_rates = [0.0, 0.2, 0.5, 0.8]

    print(f"\n[Memory Capacity vs. N and r] (rho=0.9, runs={n_runs})")
    print(f"{'Size N':<8} | " + " | ".join(f"r={r:4.1f}  " for r in retainment_rates))
    print("-" * 58)
    for n_neurons in sizes:
        row = []
        for r_val in retainment_rates:
            res = rc.parameter_sweep(
                {"retainment_rate": [r_val]},
                lambda seed, retainment_rate, n=n_neurons: rc.evaluate_memory_capacity(
                    node_type="integrator",
                    n_neurons=n,
                    max_delay=60,
                    scale_value=0.9,
                    retainment_rate=retainment_rate,
                    train_steps=800,
                    test_steps=800,
                    washout=150,
                    compute_lyapunov=False,
                    seed=seed,
                ),
                n_runs=n_runs,
                base_seed=200,
            )[0]
            row.append(f"{res['mc_mean']:8.2f}")
        print(f"{n_neurons:<8} | " + " | ".join(row))

    print(f"\n[NARMA-10 MSE vs. N and r] (rho=0.9, runs={n_runs})")
    print(f"{'Size N':<8} | " + " | ".join(f"r={r:4.1f}  " for r in retainment_rates))
    print("-" * 58)
    for n_neurons in sizes:
        row = []
        for r_val in retainment_rates:
            res = rc.parameter_sweep(
                {"retainment_rate": [r_val]},
                lambda seed, retainment_rate, n=n_neurons: rc.evaluate_narma10(
                    node_type="integrator",
                    n_neurons=n,
                    scale_value=0.9,
                    retainment_rate=retainment_rate,
                    train_steps=800,
                    test_steps=800,
                    washout=100,
                    compute_lyapunov=False,
                    seed=seed,
                ),
                n_runs=n_runs,
                base_seed=300,
            )[0]
            row.append(f"{res['mse_mean']:8.5f}")
        print(f"{n_neurons:<8} | " + " | ".join(row))


def experiment_bounds_and_pseudo_lyapunov(n_runs: int, n_neurons: int) -> None:
    """Reproduce Figs. 7 & 8: rho, mu (ssv), sigma (lsv) aligned by h_tilde_max."""
    _print_header(
        "Experiment 3 (Figs. 7 & 8): ESP Bounds & Pseudo-Lyapunov Exponent"
    )
    metric_grids = [
        ("spectral_radius", "rho (spectral radius)", [0.5, 0.8, 1.0, 1.2, 1.5]),
        ("ssv", "mu  (structured SV)", [0.8, 1.2, 1.5, 1.9, 2.4]),
        ("lsv", "sigma (largest SV)", [0.9, 1.4, 1.8, 2.2, 2.7]),
    ]

    print(f"\n[NARMA-10: Tanh Reservoir, N={n_neurons}, runs={n_runs}]")
    for metric_key, label, values in metric_grids:
        print(f"\n  Metric: {label}")
        print(f"  {'Scale':>8} | {'MSE (mean +/- std)':>22} | {'h_tilde_max':>12}")
        print("  " + "-" * 50)
        best = None
        for val in values:
            res = rc.parameter_sweep(
                {"scale_value": [val]},
                lambda seed, scale_value, mk=metric_key: rc.evaluate_narma10(
                    node_type="tanh",
                    n_neurons=n_neurons,
                    scale_metric=mk,
                    scale_value=scale_value,
                    train_steps=800,
                    test_steps=800,
                    washout=100,
                    compute_lyapunov=True,
                    lyapunov_sample_step=10,
                    seed=seed,
                ),
                n_runs=n_runs,
                base_seed=400,
            )[0]
            m_mse = res["mse_mean"]
            s_mse = res["mse_std"]
            m_h = res["pseudo_lyapunov_mean"]
            if best is None or m_mse < best[1]:
                best = (val, m_mse, m_h)
            print(f"  {val:8.2f} | {m_mse:9.5f} +/- {s_mse:8.5f} | {m_h:12.3f}")
        print(
            f"  --> Best {metric_key}: scale={best[0]:.2f}, "
            f"MSE={best[1]:.5f}, h_tilde_max={best[2]:.3f}"
        )


def experiment_synthetic_speech_spiking(n_runs: int, n_neurons: int) -> None:
    """Reproduce Appendix A.3 & B pipeline on synthetic multi-channel digits."""
    _print_header(
        "Experiment 4 (Figs. 1 & 6, Appendix A.3/B): Analog & Spiking Classification"
    )
    print(
        "(Using synthetic multi-channel cochleagram-like formant trajectories "
        "since TI46 is proprietary.)"
    )
    seqs, labels = rc.generate_synthetic_cochleagram_digits(
        n_classes=10,
        n_samples_per_class=12,
        n_channels=20,
        n_timesteps=80,
        noise_std=0.25,
        seed=7,
    )
    configs = [
        ("tanh (analog)", "tanh", None, {}, {}),
        (
            "integrator (r=0.6)",
            "integrator",
            None,
            {},
            {"retainment_rate": 0.6},
        ),
        (
            "LIF + BSA spikes",
            "lif",
            "bsa",
            {"tau": 10.0, "threshold": 0.5},
            {"tau_m": 15.0, "threshold": 0.7, "filter_tau": 20.0, "subsample": 4},
        ),
        (
            "Booij LIF + BSA (2x PSP)",
            "booij",
            "bsa",
            {"tau": 10.0, "threshold": 0.5},
            {
                "tau_m": 15.0,
                "threshold": 0.7,
                "compensate_psp": True,
                "filter_tau": 20.0,
                "subsample": 4,
            },
        ),
    ]

    print(f"\n{'Model':<26} | {'WER (mean +/- std)':>20} | {'Accuracy':>10}")
    print("-" * 64)
    for label, node_type, enc, enc_kw, node_kw in configs:
        wers = []
        for r in range(n_runs):
            out = rc.evaluate_classification_cv(
                sequences=seqs,
                labels=labels,
                node_type=node_type,
                n_neurons=n_neurons,
                n_folds=4,
                scale_value=0.9,
                spike_encoding=enc,
                encoding_kwargs=enc_kw,
                regularization=1e-3,
                seed=500 + r,
                **node_kw,
            )
            wers.append(out["wer"])
        mean_wer = float(np.mean(wers))
        std_wer = float(np.std(wers))
        print(
            f"{label:<26} | {mean_wer:8.3f} +/- {std_wer:7.3f} | "
            f"{(1.0 - mean_wer) * 100.0:9.1f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from Verstraeten et al. (2007)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of stochastic reservoir realizations per setting (default: 3).",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=60,
        help="Default reservoir size for sweeps (default: 60).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast configuration (1 run, N=40).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_runs = 1 if args.quick else args.runs
    n_neurons = 40 if args.quick else args.n_neurons

    print("Reservoir Computing Unification Demo (Verstraeten et al., 2007)")
    print(f"Configuration: runs={n_runs}, n_neurons={n_neurons}")

    experiment_node_complexity(n_runs=n_runs, n_neurons=n_neurons)
    experiment_node_vs_reservoir_memory(n_runs=n_runs)
    experiment_bounds_and_pseudo_lyapunov(n_runs=n_runs, n_neurons=n_neurons)
    experiment_synthetic_speech_spiking(n_runs=n_runs, n_neurons=n_neurons)
    print("\nDone.")


if __name__ == "__main__":
    main()
