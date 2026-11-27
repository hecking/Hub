"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from hub.reservoir.metrics import largest_singular_value, spectral_radius
from hub.reservoir.topology import (
    assign_weights,
    generate_input_weights,
    generate_reservoir_weights,
    lattice_connectivity,
    lsm_3d_connectivity,
    random_connectivity,
)


class TestTopology(unittest.TestCase):
    def test_random_connectivity_density(self):
        mask = random_connectivity(100, connection_fraction=0.5, rng=0)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, bool)
        density = float(np.mean(mask))
        self.assertGreater(density, 0.42)
        self.assertLess(density, 0.58)

    def test_lattice_2d_and_3d_with_rewiring(self):
        mask_2d = lattice_connectivity((5, 5), rewire_prob=0.0, periodic=True)
        self.assertEqual(mask_2d.shape, (25, 25))
        # Each node in a periodic 5x5 lattice has degree 4.
        np.testing.assert_array_equal(np.sum(mask_2d, axis=0), np.full(25, 4))

        mask_3d = lattice_connectivity((3, 3, 3), rewire_prob=0.0, periodic=True)
        self.assertEqual(mask_3d.shape, (27, 27))
        np.testing.assert_array_equal(np.sum(mask_3d, axis=0), np.full(27, 6))

        mask_sw = lattice_connectivity((5, 5), rewire_prob=0.3, periodic=True, rng=42)
        self.assertFalse(np.array_equal(mask_2d, mask_sw))

    def test_lsm_3d_distance_decay(self):
        mask_short = lsm_3d_connectivity((4, 4, 4), c=0.5, lam=1.0, rng=1)
        mask_long = lsm_3d_connectivity((4, 4, 4), c=0.5, lam=3.0, rng=1)
        self.assertEqual(mask_short.shape, (64, 64))
        self.assertEqual(np.sum(np.diag(mask_short)), 0)
        self.assertGreater(np.sum(mask_long), np.sum(mask_short))

    def test_assign_weights_distributions(self):
        mask = random_connectivity(30, connection_fraction=0.5, rng=0)
        w_norm = assign_weights(mask, distribution="normal", rng=1)
        self.assertTrue(np.all(w_norm[~mask] == 0.0))
        self.assertGreater(np.std(w_norm[mask]), 0.5)

        w_unif = assign_weights(mask, distribution="uniform", rng=2)
        self.assertTrue(np.all(w_unif[mask] >= -1.0))
        self.assertTrue(np.all(w_unif[mask] <= 1.0))

        w_disc = assign_weights(
            mask, distribution="discrete", discrete_values=(-0.5, 0.5), rng=3
        )
        self.assertEqual(set(np.unique(w_disc[mask])), {-0.5, 0.5})

    def test_generate_reservoir_and_input_weights(self):
        for topo in ("random", "lattice_2d", "lattice_3d", "lsm_3d"):
            w = generate_reservoir_weights(
                n_neurons=27,
                topology=topo,
                scale_metric="spectral_radius",
                scale_value=0.85,
                rng=10,
            )
            self.assertEqual(w.shape, (27, 27))
            self.assertAlmostEqual(spectral_radius(w), 0.85, places=4)

        w_lsv = generate_reservoir_weights(
            n_neurons=25, scale_metric="lsv", scale_value=1.3, rng=11
        )
        self.assertAlmostEqual(largest_singular_value(w_lsv), 1.3, places=4)

        w_in = generate_input_weights(
            n_neurons=100, n_inputs=3, connection_fraction=0.1, rng=5
        )
        self.assertEqual(w_in.shape, (100, 3))
        density = float(np.mean(w_in != 0.0))
        self.assertGreater(density, 0.03)
        self.assertLess(density, 0.25)


if __name__ == "__main__":
    unittest.main()
