"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import unittest

import numpy as np

from hub.reservoir.encoding import (
    bsa_decode,
    bsa_encode,
    exponential_filter_kernel,
    exponential_lowpass_filter,
    lif_encode,
    poisson_encode,
)
from hub.reservoir.metrics import mse


class TestEncoding(unittest.TestCase):
    def test_poisson_encoding_rate(self):
        sig = np.full((1000, 2), [0.1, 0.8])
        spikes = poisson_encode(sig, max_rate=1.0, dt=1.0, rng=0)
        self.assertEqual(spikes.shape, sig.shape)
        self.assertTrue(set(np.unique(spikes)).issubset({0.0, 1.0}))
        self.assertLess(np.mean(spikes[:, 0]), 0.2)
        self.assertGreater(np.mean(spikes[:, 1]), 0.7)

    def test_lif_encoding(self):
        low = np.full(200, 0.05)
        high = np.full(200, 0.4)
        spk_low = lif_encode(low, tau_m=20.0, threshold=1.0)
        spk_high = lif_encode(high, tau_m=20.0, threshold=1.0)
        self.assertEqual(spk_low.shape, (200,))
        self.assertGreater(np.sum(spk_high), np.sum(spk_low))

    def test_bsa_encode_decode_reconstruction(self):
        t = np.linspace(0.0, 4.0 * np.pi, 400)
        sig = np.column_stack(
            [0.5 + 0.4 * np.sin(t), 0.5 + 0.35 * np.cos(0.7 * t)]
        )
        spikes = bsa_encode(sig, tau=12.0, threshold=0.5)
        self.assertEqual(spikes.shape, sig.shape)
        self.assertTrue(set(np.unique(spikes)).issubset({0.0, 1.0}))
        self.assertGreater(np.sum(spikes), 50.0)

        recon = bsa_decode(spikes, tau=12.0)
        # Exclude initial filter transient.
        self.assertLess(mse(recon[40:], sig[40:]), 0.025)

    def test_exponential_filter_and_lowpass_subsampling(self):
        kernel = exponential_filter_kernel(tau=30.0)
        self.assertAlmostEqual(float(np.sum(kernel)), 1.0, places=6)
        self.assertGreater(kernel[0], kernel[-1])

        spikes = np.zeros((100, 3))
        spikes[0, 0] = 1.0
        spikes[10, 1] = 1.0
        filtered = exponential_lowpass_filter(spikes, tau=20.0, subsample=5)
        self.assertEqual(filtered.shape, (20, 3))
        self.assertGreater(filtered[0, 0], filtered[5, 0])
        self.assertAlmostEqual(filtered[0, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
