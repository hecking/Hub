"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from hub.reservoir.benchmarks import (
    generate_mackey_glass,
    generate_memory_capacity_data,
    generate_narma10,
    generate_synthetic_cochleagram_digits,
)
from hub.reservoir.metrics import mse
from hub.reservoir.nodes import create_reservoir
from hub.reservoir.pipeline import (
    ReservoirPipeline,
    evaluate_classification_cv,
    evaluate_memory_capacity,
    evaluate_narma10,
    parameter_sweep,
)
from hub.reservoir.readout import (
    BPDCNetwork,
    LinearReadout,
    PostProcessingPipeline,
    speech_classification_pipeline,
)


class TestReadoutAndPipeline(unittest.TestCase):
    def test_linear_readout_pinv_and_ridge(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((200, 5))
        true_w = np.array([1.5, -2.0, 0.5, 0.0, 3.0])
        y = x @ true_w + 0.7

        ro_pinv = LinearReadout(regularization=0.0).fit(x, y)
        self.assertLess(mse(ro_pinv.predict(x), y), 1e-20)

        ro_ridge = LinearReadout(regularization=1e-4).fit(x, y)
        self.assertLess(mse(ro_ridge.predict(x), y), 1e-8)

    def test_post_processing_pipeline(self):
        raw = np.array(
            [
                [0.1, 0.9, -0.2],
                [0.2, 0.8, -0.1],
                [0.0, 0.7, -0.3],
            ]
        )
        speech_pp = speech_classification_pipeline()
        self.assertEqual(speech_pp(raw), 1)

        custom_pp = PostProcessingPipeline(
            nonlinearity1="tanh",
            temporal_filter="lowpass",
            nonlinearity2="sign",
            lowpass_tau=5.0,
        )
        out = custom_pp(raw)
        self.assertEqual(out.shape, raw.shape)
        self.assertTrue(set(np.unique(out)).issubset({-1.0, 0.0, 1.0}))

    def test_benchmarks_shapes_and_finiteness(self):
        u_n, y_n = generate_narma10(n_steps=500, seed=1)
        self.assertEqual(u_n.shape, (500, 1))
        self.assertEqual(y_n.shape, (500,))
        self.assertTrue(np.all(np.isfinite(y_n)))

        u_m, y_m = generate_memory_capacity_data(
            n_steps=400, max_delay=25, seed=2
        )
        self.assertEqual(u_m.shape, (400, 1))
        self.assertEqual(y_m.shape, (400, 25))
        # Delay 1 target at t equals u at t - 1.
        np.testing.assert_allclose(y_m[1:, 0], u_m[:-1, 0])

        mg = generate_mackey_glass(n_steps=300, tau=17, seed=3)
        self.assertEqual(mg.shape, (300,))
        self.assertTrue(np.all(np.isfinite(mg)))
        self.assertGreater(float(np.std(mg)), 0.05)

        seqs, labels = generate_synthetic_cochleagram_digits(
            n_classes=4, n_samples_per_class=5, n_channels=12, n_timesteps=40, seed=4
        )
        self.assertEqual(len(seqs), 20)
        self.assertEqual(seqs[0].shape, (40, 12))
        self.assertEqual(set(labels.tolist()), {0, 1, 2, 3})

    def test_bpdc_learns_narma10(self):
        u_tr, y_tr = generate_narma10(n_steps=900, seed=11)
        u_te, y_te = generate_narma10(n_steps=600, seed=12)

        net = BPDCNetwork(
            n_neurons=40,
            n_inputs=1,
            n_outputs=1,
            scale_value=0.9,
            learning_rate=0.05,
            leak_rate=0.8,
            feedback_scale=0.1,
            rng=42,
        )
        net.fit(u_tr, y_tr, washout=100, epochs=5)
        preds = net.predict(u_te, reset=True)
        self.assertEqual(preds.shape, y_te.shape)
        self.assertTrue(np.all(np.isfinite(preds)))
        # Must beat trivial mean baseline.
        baseline_mse = float(np.var(y_te[100:]))
        self.assertLess(mse(preds[100:], y_te[100:]), baseline_mse)

    def test_paper_qualitative_findings_end_to_end(self):
        # Finding 1 (Fig. 2): linear nodes achieve highest memory capacity,
        # and analog integrator (r > 0) has lower MC than plain tanh.
        mc_lin = evaluate_memory_capacity(
            node_type="linear",
            n_neurons=40,
            max_delay=50,
            scale_value=0.9,
            train_steps=600,
            test_steps=600,
            washout=100,
            seed=7,
        )["mc"]
        mc_tanh = evaluate_memory_capacity(
            node_type="tanh",
            n_neurons=40,
            max_delay=50,
            scale_value=0.9,
            train_steps=600,
            test_steps=600,
            washout=100,
            seed=7,
        )["mc"]
        mc_int = evaluate_memory_capacity(
            node_type="integrator",
            n_neurons=40,
            max_delay=50,
            scale_value=0.9,
            retainment_rate=0.5,
            train_steps=600,
            test_steps=600,
            washout=100,
            seed=7,
        )["mc"]
        self.assertGreater(mc_lin, mc_tanh)
        self.assertGreater(mc_tanh, mc_int)

        # Finding 2 (Fig. 3): NARMA-10 tanh reservoir attains low MSE at rho=0.9.
        narma_res = evaluate_narma10(
            node_type="tanh",
            n_neurons=50,
            scale_value=0.9,
            train_steps=800,
            test_steps=800,
            washout=100,
            seed=7,
        )
        self.assertLess(narma_res["mse"], 0.004)
        self.assertIn("pseudo_lyapunov", narma_res)

    def test_pipeline_classification_and_parameter_sweep(self):
        seqs, labels = generate_synthetic_cochleagram_digits(
            n_classes=4, n_samples_per_class=8, n_channels=12, n_timesteps=50, seed=0
        )
        res_analog = evaluate_classification_cv(
            seqs, labels, node_type="tanh", n_neurons=30, n_folds=4, seed=1
        )
        self.assertLess(res_analog["wer"], 0.25)

        res_spiking = evaluate_classification_cv(
            seqs,
            labels,
            node_type="booij",
            n_neurons=30,
            n_folds=4,
            spike_encoding="bsa",
            encoding_kwargs={"tau": 10.0, "threshold": 0.5},
            tau_m=15.0,
            threshold=0.7,
            filter_tau=20.0,
            subsample=4,
            seed=1,
        )
        self.assertLess(res_spiking["wer"], 0.35)

        sweep = parameter_sweep(
            {"scale_value": [0.5, 0.9]},
            lambda seed, scale_value: evaluate_narma10(
                node_type="tanh",
                n_neurons=25,
                scale_value=scale_value,
                train_steps=400,
                test_steps=400,
                washout=80,
                seed=seed,
            ),
            n_runs=2,
            base_seed=0,
        )
        self.assertEqual(len(sweep), 2)
        self.assertIn("mse_mean", sweep[0])
        self.assertIn("mse_std", sweep[0])
        self.assertIn("pseudo_lyapunov_mean", sweep[0])

    def test_reservoir_pipeline_direct_usage(self):
        u_tr, y_tr = generate_narma10(n_steps=500, seed=21)
        u_te, y_te = generate_narma10(n_steps=400, seed=22)
        res = create_reservoir("tanh", n_neurons=35, n_inputs=1, rng=21)
        pipe = ReservoirPipeline(reservoir=res, washout=80)
        pipe.fit(u_tr, y_tr)
        preds = pipe.predict(u_te)
        self.assertEqual(preds.shape, y_te.shape)
        self.assertLess(mse(preds[80:], y_te[80:]), 0.005)


if __name__ == "__main__":
    unittest.main()
