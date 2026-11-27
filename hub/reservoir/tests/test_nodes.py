"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from hub.reservoir.nodes import (
    AnalogIntegratorReservoir,
    BooijLIFReservoir,
    EventBasedBooijReservoir,
    LIFReservoir,
    LinearReservoir,
    TanhReservoir,
    booij_next_spike_time,
    create_reservoir,
)
from hub.reservoir.topology import (
    generate_input_weights,
    generate_reservoir_weights,
)


class TestNodes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.w = generate_reservoir_weights(20, scale_value=0.9, rng=rng)
        self.w_in = generate_input_weights(
            20, 2, connection_fraction=0.5, rng=rng
        )
        self.u = rng.uniform(-0.5, 0.5, size=(100, 2))

    def test_linear_and_tanh_reservoirs(self):
        lin = LinearReservoir(self.w, self.w_in)
        st_lin = lin.run(self.u)
        self.assertEqual(st_lin.shape, (100, 20))
        np.testing.assert_allclose(st_lin[0], self.w_in @ self.u[0])

        tn = TanhReservoir(self.w, self.w_in)
        st_tn, pre_tn = tn.run(self.u, return_pre_activations=True)
        self.assertEqual(st_tn.shape, (100, 20))
        np.testing.assert_allclose(st_tn, np.tanh(pre_tn))

    def test_analog_integrator_reduces_to_tanh_at_zero_retainment(self):
        tn = TanhReservoir(self.w, self.w_in)
        integ0 = AnalogIntegratorReservoir(
            self.w, self.w_in, retainment_rate=0.0
        )
        np.testing.assert_allclose(tn.run(self.u), integ0.run(self.u))

        integ_tau = AnalogIntegratorReservoir(
            self.w, self.w_in, tau_int=5.0, dt=1.0
        )
        self.assertAlmostEqual(integ_tau.retainment_rate, 0.8)
        st_slow = integ_tau.run(self.u)
        st_fast = integ0.run(self.u)
        # Higher retainment rate produces smoother (smaller step-to-step diff) dynamics.
        self.assertLess(
            float(np.mean(np.abs(np.diff(st_slow, axis=0)))),
            float(np.mean(np.abs(np.diff(st_fast, axis=0)))),
        )

    def test_lif_and_booij_spiking_reservoirs(self):
        rng = np.random.default_rng(4)
        spk_in = (rng.random((120, 2)) < 0.35).astype(np.float64)
        w_pos = np.abs(self.w) * 0.25
        w_in_pos = np.abs(self.w_in) * 0.8

        lif = LIFReservoir(
            w_pos, w_in_pos, tau_m=15.0, threshold=1.0, filter_tau=20.0, subsample=4
        )
        raw_lif = lif.run_spikes(spk_in)
        filt_lif = lif.run(spk_in)
        self.assertEqual(raw_lif.shape, (120, 20))
        self.assertEqual(filt_lif.shape, (30, 20))
        self.assertGreater(float(np.sum(raw_lif)), 0.0)

        booij_comp = BooijLIFReservoir(
            w_pos,
            w_in_pos,
            tau_m=15.0,
            threshold=1.0,
            compensate_psp=True,
            filter_tau=20.0,
            subsample=4,
        )
        booij_nocomp = BooijLIFReservoir(
            w_pos,
            w_in_pos,
            tau_m=15.0,
            threshold=1.0,
            compensate_psp=False,
            filter_tau=20.0,
            subsample=4,
        )
        self.assertEqual(booij_comp.psp_scale, 2.0)
        self.assertEqual(booij_nocomp.psp_scale, 1.0)
        spk_comp = booij_comp.run_spikes(spk_in)
        spk_nocomp = booij_nocomp.run_spikes(spk_in)
        # Appendix B: PSP compensation restores activity vs uncompensated Booij.
        self.assertGreaterEqual(float(np.sum(spk_comp)), float(np.sum(spk_nocomp)))
        self.assertEqual(booij_comp.run(spk_in).shape, (30, 20))

    def test_booij_quadratic_next_spike_time_exact(self):
        tau_m = 20.0
        theta = 1.0
        # Incoming excitatory jump a = b = 5.0 gives peak 5^2 / (4*5) = 1.25 > 1.0.
        dt_spike = booij_next_spike_time(a=5.0, b=5.0, tau_m=tau_m, threshold=theta)
        self.assertIsNotNone(dt_spike)
        self.assertGreater(dt_spike, 0.0)

        # Verify exact threshold crossing u(dt_spike) == theta.
        u_at_spike = 5.0 * np.exp(-dt_spike / tau_m) - 5.0 * np.exp(
            -2.0 * dt_spike / tau_m
        )
        self.assertAlmostEqual(u_at_spike, theta, places=10)

        # Verify u(t) < theta strictly before dt_spike (earliest root).
        grid = np.linspace(0.0, dt_spike * 0.999, 200)
        u_before = 5.0 * np.exp(-grid / tau_m) - 5.0 * np.exp(-2.0 * grid / tau_m)
        self.assertTrue(np.all(u_before < theta))

        # Sub-threshold peak (a = b = 3.0 -> peak 0.75 < 1.0) must return None.
        self.assertIsNone(
            booij_next_spike_time(a=3.0, b=3.0, tau_m=tau_m, threshold=theta)
        )

    def test_event_based_booij_reservoir(self):
        w = np.zeros((4, 4))
        w[1, 0] = 2.5
        w[2, 1] = 2.5
        w_in = np.zeros((4, 1))
        w_in[0, 0] = 3.0

        res = EventBasedBooijReservoir(
            weights=w,
            input_weights=w_in,
            tau_m=10.0,
            threshold=1.0,
            synaptic_delay=1.0,
            compensate_psp=True,
            dt=1.0,
            filter_tau=10.0,
        )
        events = res.simulate_events([(0.0, 0)], t_end=30.0)
        # Spike should propagate 0 -> 1 -> 2.
        self.assertGreater(len(events[0]), 0)
        self.assertGreater(len(events[1]), 0)
        self.assertGreater(len(events[2]), 0)
        self.assertGreater(events[1][0], events[0][0] + 1.0)

        spk_in = np.zeros((30, 1))
        spk_in[0, 0] = 1.0
        out = res.run(spk_in)
        self.assertEqual(out.shape, (30, 4))
        self.assertGreater(float(np.sum(out)), 0.0)

    def test_factory_creates_all_node_types(self):
        for nt in ("linear", "tanh", "integrator", "lif", "booij", "booij_event"):
            res = create_reservoir(node_type=nt, n_neurons=10, n_inputs=2, rng=0)
            self.assertEqual(res.n_neurons, 10)
            self.assertEqual(res.n_inputs, 2)


if __name__ == "__main__":
    unittest.main()
