"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from hub.reservoir.metrics import rescale_weights


def _get_rng(
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def random_connectivity(
    n_neurons: int,
    connection_fraction: float = 0.5,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate a random binary connectivity matrix with given connection fraction.

    Default connection_fraction=0.5 matches Verstraeten et al. (2007, Appendix A).
    """
    if n_neurons < 1:
        raise ValueError("n_neurons must be >= 1.")
    if not 0.0 <= connection_fraction <= 1.0:
        raise ValueError("connection_fraction must be in [0, 1].")
    gen = _get_rng(rng)
    mask = gen.random((n_neurons, n_neurons)) < connection_fraction
    if connection_fraction > 0.0 and not np.any(mask):
        idx = int(gen.integers(0, n_neurons))
        mask[idx, idx] = True
    return mask


def _infer_lattice_shape(n_neurons: int, dim: int) -> Tuple[int, ...]:
    """Infer a near-cubic/square lattice shape whose product equals n_neurons."""
    if dim == 2:
        side = int(np.floor(np.sqrt(n_neurons)))
        while side > 1 and n_neurons % side != 0:
            side -= 1
        return (side, n_neurons // side)
    if dim == 3:
        s1 = int(round(n_neurons ** (1.0 / 3.0)))
        while s1 > 1 and n_neurons % s1 != 0:
            s1 -= 1
        rem = n_neurons // s1
        s2 = int(np.floor(np.sqrt(rem)))
        while s2 > 1 and rem % s2 != 0:
            s2 -= 1
        return (s1, s2, rem // s2)
    raise ValueError("dim must be 2 or 3.")


def lattice_connectivity(
    shape: Tuple[int, ...],
    rewire_prob: float = 0.0,
    periodic: bool = True,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate a 2D or 3D regular lattice with optional Watts-Strogatz rewiring.

    Implements the regular lattice and small-world connectivity generation
    described in Verstraeten et al. (2007, Section 3.2; Watts & Strogatz, 1998).
    """
    if len(shape) not in (2, 3) or any(s < 1 for s in shape):
        raise ValueError("shape must be a 2-tuple or 3-tuple of positive integers.")
    if not 0.0 <= rewire_prob <= 1.0:
        raise ValueError("rewire_prob must be in [0, 1].")

    n_neurons = int(np.prod(shape))
    mask = np.zeros((n_neurons, n_neurons), dtype=bool)
    ndim = len(shape)
    coords = np.array(np.unravel_index(np.arange(n_neurons), shape)).T

    for idx in range(n_neurons):
        c = coords[idx]
        for axis in range(ndim):
            for delta in (-1, 1):
                nb = c.copy()
                nb[axis] += delta
                if periodic:
                    nb[axis] %= shape[axis]
                elif nb[axis] < 0 or nb[axis] >= shape[axis]:
                    continue
                nb_idx = int(np.ravel_multi_index(tuple(nb), shape))
                if nb_idx != idx:
                    mask[nb_idx, idx] = True

    if rewire_prob > 0.0 and n_neurons > 1:
        gen = _get_rng(rng)
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows, cols):
            if gen.random() < rewire_prob:
                mask[r, c] = False
                new_r = int(gen.integers(0, n_neurons))
                mask[new_r, c] = True

    return mask


def lsm_3d_connectivity(
    shape: Tuple[int, int, int],
    c: float = 0.3,
    lam: float = 2.0,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate a 3D LSM microcolumn connectivity mask (Maass et al., 2002).

    Connection probability between nodes a and b on a 3D lattice is given by
    Verstraeten et al. (2007, Section 3.2):

        P_conn(a, b) = C * exp(-D^2(a, b) / lambda^2)
    """
    if len(shape) != 3 or any(s < 1 for s in shape):
        raise ValueError("shape must be a 3-tuple of positive integers.")
    if c < 0.0 or lam <= 0.0:
        raise ValueError("c must be >= 0 and lam must be > 0.")

    gen = _get_rng(rng)
    n_neurons = int(np.prod(shape))
    coords = np.array(
        np.unravel_index(np.arange(n_neurons), shape), dtype=np.float64
    ).T
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    probs = np.clip(c * np.exp(-dist_sq / (lam ** 2)), 0.0, 1.0)
    np.fill_diagonal(probs, 0.0)
    mask = gen.random((n_neurons, n_neurons)) < probs
    if c > 0.0 and not np.any(mask) and n_neurons > 1:
        mask[0, 1] = True
    return mask


def assign_weights(
    mask: np.ndarray,
    distribution: str = "normal",
    discrete_values: Sequence[float] = (-1.0, 1.0),
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Assign random weights to nonzero entries of a connectivity mask.

    Supports 'normal' (standard Gaussian, paper default), 'uniform' ([-1, 1]),
    and 'discrete' (uniformly drawn from discrete_values).
    """
    mask_bool = np.asarray(mask, dtype=bool)
    gen = _get_rng(rng)
    w = np.zeros(mask_bool.shape, dtype=np.float64)
    n_nonzero = int(np.sum(mask_bool))
    if n_nonzero == 0:
        return w

    dist = distribution.lower()
    if dist == "normal":
        vals = gen.standard_normal(n_nonzero)
    elif dist == "uniform":
        vals = gen.uniform(-1.0, 1.0, size=n_nonzero)
    elif dist == "discrete":
        choices = np.asarray(discrete_values, dtype=np.float64)
        if choices.size == 0:
            raise ValueError("discrete_values must be non-empty.")
        vals = gen.choice(choices, size=n_nonzero)
    else:
        raise ValueError(
            f"Unsupported weight distribution '{distribution}'. "
            "Expected 'normal', 'uniform', or 'discrete'."
        )
    w[mask_bool] = vals
    return w


def _build_connectivity_mask(
    n_neurons: int,
    topology: str,
    connection_fraction: float,
    lattice_shape: Optional[Tuple[int, ...]],
    rewire_prob: float,
    lsm_c: float,
    lsm_lambda: float,
    gen: np.random.Generator,
) -> np.ndarray:
    topo = topology.lower()
    if topo == "random":
        return random_connectivity(
            n_neurons, connection_fraction=connection_fraction, rng=gen
        )
    if topo == "lattice_2d":
        shape = (
            lattice_shape
            if lattice_shape is not None
            else _infer_lattice_shape(n_neurons, 2)
        )
        return lattice_connectivity(shape, rewire_prob=rewire_prob, rng=gen)
    if topo == "lattice_3d":
        shape = (
            lattice_shape
            if lattice_shape is not None
            else _infer_lattice_shape(n_neurons, 3)
        )
        return lattice_connectivity(shape, rewire_prob=rewire_prob, rng=gen)
    if topo in ("lsm_3d", "microcolumn"):
        shape = (
            lattice_shape
            if lattice_shape is not None
            else _infer_lattice_shape(n_neurons, 3)
        )
        return lsm_3d_connectivity(shape, c=lsm_c, lam=lsm_lambda, rng=gen)
    raise ValueError(
        f"Unsupported topology '{topology}'. Expected 'random', "
        "'lattice_2d', 'lattice_3d', or 'lsm_3d'."
    )


def generate_reservoir_weights(
    n_neurons: int,
    topology: str = "random",
    connection_fraction: float = 0.5,
    weight_distribution: str = "normal",
    scale_metric: Optional[str] = "spectral_radius",
    scale_value: float = 0.9,
    lattice_shape: Optional[Tuple[int, ...]] = None,
    rewire_prob: float = 0.0,
    lsm_c: float = 0.3,
    lsm_lambda: float = 2.0,
    discrete_values: Sequence[float] = (-1.0, 1.0),
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate a reservoir weight matrix via the 3-stage pipeline (Section 3.2).

    1. Determine connectivity topology ('random', 'lattice_2d', 'lattice_3d', 'lsm_3d').
    2. Assign weights ('normal', 'uniform', 'discrete').
    3. Rescale global matrix by scale_metric ('spectral_radius', 'lsv', 'ssv', or None).
    """
    gen = _get_rng(rng)
    for _ in range(10):
        mask = _build_connectivity_mask(
            n_neurons=n_neurons,
            topology=topology,
            connection_fraction=connection_fraction,
            lattice_shape=lattice_shape,
            rewire_prob=rewire_prob,
            lsm_c=lsm_c,
            lsm_lambda=lsm_lambda,
            gen=gen,
        )
        w = assign_weights(
            mask,
            distribution=weight_distribution,
            discrete_values=discrete_values,
            rng=gen,
        )
        if scale_metric is None:
            return w
        try:
            return rescale_weights(w, target=scale_value, metric=scale_metric)
        except ValueError:
            continue
    raise RuntimeError(
        "Failed to generate a reservoir weight matrix with non-zero "
        f"{scale_metric} after 10 attempts."
    )


def generate_input_weights(
    n_neurons: int,
    n_inputs: int,
    connection_fraction: float = 0.1,
    weight_distribution: str = "normal",
    input_scale: float = 1.0,
    discrete_values: Sequence[float] = (-1.0, 1.0),
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate reservoir input weight matrix W_in of shape (n_neurons, n_inputs).

    Defaults follow Verstraeten et al. (2007, Appendix A): connection fraction
    0.1, normal distribution, left unscaled (input_scale=1.0).
    """
    if n_neurons < 1 or n_inputs < 1:
        raise ValueError("n_neurons and n_inputs must be >= 1.")
    if not 0.0 <= connection_fraction <= 1.0:
        raise ValueError("connection_fraction must be in [0, 1].")
    gen = _get_rng(rng)
    mask = gen.random((n_neurons, n_inputs)) < connection_fraction
    if connection_fraction > 0.0 and not np.any(mask):
        mask[int(gen.integers(0, n_neurons)), 0] = True
    w_in = assign_weights(
        mask,
        distribution=weight_distribution,
        discrete_values=discrete_values,
        rng=gen,
    )
    return w_in * float(input_scale)
