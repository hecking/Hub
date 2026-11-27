"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from hub.reservoir.metrics import (
    largest_singular_value,
    max_lyapunov_exponent,
    memory_capacity,
    mse,
    nrmse,
    pseudo_lyapunov_exponent,
    rescale_weights,
    reservoir_jacobian,
    spectral_radius,
    structured_singular_value,
    word_error_rate,
)
from hub.reservoir.nodes import AnalogIntegratorReservoir, TanhReservoir
from hub.reservoir.topology import (
    generate_input_weights,
    generate_reservoir_weights,
)


class TestMetrics(unittest.TestCase):
    def test_stability_bound_ordering_and_rescaling(self):
        rng = np.random.default_rng(123)
        w = rng.standard_normal((30, 30)) * (rng.random((30, 30)) < 0.5)

        rho = spectral_radius(w)
        mu = structured_singular_value(w)
        sigma = largest_singular_value(w)

        self.assertGreater(rho, 0.0)
        self.assertLessEqual(rho, mu + 1e-4)
        self.assertLessEqual(mu, sigma + 1e-4)

        # Diagonal similarity invariance of rho and mu (unlike sigma).
        d = np.exp(rng.uniform(-1.5, 1.5, size=30))
        w_sim = (d[:, None] * w) / d[None, :]
        self.assertAlmostEqual(spectral_radius(w_sim), rho, places=5)
        self.assertAlmostEqual(
            structured_singular_value(w_sim), mu, delta=0.15 * mu
        )
        self.assertGreater(largest_singular_value(w_sim), sigma)

        for metric, fn in (
            ("spectral_radius", spectral_radius),
            ("lsv", largest_singular_value),
            ("ssv", structured_singular_value),
        ):
            w_scaled = rescale_weights(w, target=1.25, metric=metric)
            self.assertAlmostEqual(fn(w_scaled), 1.25, delta=1e-2)

    def test_analytical_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(7)
        n = 12
        w = generate_reservoir_weights(n, scale_value=0.9, rng=rng)
        w_in = generate_input_weights(n, 2, connection_fraction=0.5, rng=rng)
        s0 = rng.uniform(-0.5, 0.5, size=n)
        u0 = rng.uniform(-0.5, 0.5, size=2)
        eps = 1e-6

        # Tanh reservoir Jacobian
        z = w_in @ u0 + w @ s0
        s1 = np.tanh(z)
        j_analytic = reservoir_jacobian(w, s1, node_type="tanh")
        j_num = np.zeros((n, n))
        for j in range(n):
            ds = np.zeros(n)
            ds[j] = eps
            f_plus = np.tanh(w_in @ u0 + w @ (s0 + ds))
            f_minus = np.tanh(w_in @ u0 + w @ (s0 - ds))
            j_num[:, j] = (f_plus - f_minus) / (2.0 * eps)
        np.testing.assert_allclose(j_analytic, j_num, atol=1e-6)

        # Analog integrator Jacobian (Eq. 1)
        r = 0.35
        s1_int = r * s0 + (1.0 - r) * np.tanh(z)
        j_int = reservoir_jacobian(
            w, s1_int, node_type="integrator", retainment_rate=r, pre_activation=z
        )
        j_int_num = np.zeros((n, n))
        for j in range(n):
            ds = np.zeros(n)
            ds[j] = eps
            f_plus = r * (s0 + ds) + (1.0 - r) * np.tanh(w_in @ u0 + w @ (s0 + ds))
            f_minus = r * (s0 - ds) + (1.0 - r) * np.tanh(w_in @ u0 + w @ (s0 - ds))
            j_int_num[:, j] = (f_plus - f_minus) / (2.0 * eps)
        np.testing.assert_allclose(j_int, j_int_num, atol=1e-6)

    def test_pseudo_lyapunov_monotonicity_and_integrator_recovery(self):
        rng = np.random.default_rng(42)
        n = 30
        w_base = generate_reservoir_weights(n, scale_value=1.0, rng=rng)
        w_in = generate_input_weights(n, 1, rng=rng)
        u = rng.uniform(-0.8, 0.8, size=(400, 1))

        h_vals = []
        for rho in (0.4, 0.9, 1.5):
            w = w_base * rho
            res = TanhReservoir(w, w_in)
            states = res.run(u)
            h_max, spec = pseudo_lyapunov_exponent(
                w,
                states,
                node_type="tanh",
                sample_step=10,
                washout=50,
                return_spectrum=True,
            )
            self.assertEqual(spec.shape, (n,))
            h_vals.append(h_max)
            h_true = max_lyapunov_exponent(
                w, states, node_type="tanh", washout=50, rng=rng
            )
            self.assertTrue(np.isfinite(h_true))

        # Local expansion grows monotonically with global scaling.
        self.assertLess(h_vals[0], h_vals[1])
        self.assertLess(h_vals[1], h_vals[2])

        # Recovered integrator pre-activations match exact pre-activations.
        res_int = AnalogIntegratorReservoir(w_base * 0.9, w_in, retainment_rate=0.4)
        st_int, pre_int = res_int.run(u, return_pre_activations=True)
        h_exact = pseudo_lyapunov_exponent(
            res_int.weights,
            st_int,
            node_type="integrator",
            retainment_rate=0.4,
            washout=50,
            pre_activations=pre_int,
        )
        h_rec = pseudo_lyapunov_exponent(
            res_int.weights,
            st_int,
            node_type="integrator",
            retainment_rate=0.4,
            washout=50,
        )
        self.assertAlmostEqual(h_exact, h_rec, places=4)

    def test_scoring_functions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mse(y, y), 0.0)
        self.assertAlmostEqual(nrmse(y, y), 0.0)
        self.assertAlmostEqual(mse(y + 1.0, y), 1.0)

        rng = np.random.default_rng(0)
        sig = rng.uniform(-1.0, 1.0, size=(500, 3))
        total_mc, per_delay = memory_capacity(sig, sig, return_per_delay=True)
        self.assertAlmostEqual(total_mc, 3.0, places=6)
        np.testing.assert_allclose(per_delay, np.ones(3), atol=1e-6)

        self.assertAlmostEqual(
            word_error_rate(np.array([0, 1, 2, 3]), np.array([0, 1, 0, 3])),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
