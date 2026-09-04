import os

import jax.numpy as jnp
import numpy as np
from matplotlib import ticker
from scipy.constants import Boltzmann

from classical_diffusion.analysis import plot_isf
from classical_diffusion.hopping import (
    Lattice1D,
    get_deterministic_isf,
    get_deterministic_probabilities,
)
from classical_diffusion.jax.langevin import (
    KramersParameters,
    KramersSystem1D,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.util import timed

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import pickle  # ruff: ignore[suspicious-pickle-import]
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import equinox as eqx
import jax

from classical_diffusion.simulation import SimulationResult, TimeSpan

jax.config.update("jax_enable_x64", False)


if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem


CHECK_EVERY = 5
EARLY_STOP = 0.005  # Improvement to loss over CHECK_EVERY epochs deemed small enough to have reached training plateau
NUM_EPOCHS = 100
BATCH_SIZE = 10  # Number of isfs to test on at a time (does this need to be limited?)
LOSS_BATCH_SIZE = 5  # Number of isfs to calculate loss of at a time


class JaxEnsembleResults(TypedDict):
    """Jax compatible output from an ensemble run."""

    parameters: KramersParameters
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating[Any]]],
        np.ndarray[Any, np.dtype[np.floating[Any]]],
    ]
    result: tuple[jnp.ndarray, jnp.ndarray]


class ResidualBlock(eqx.Module):
    """Residual Block in ResNet architecture. H(x) = F(x) + x."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, *, key: jax.Array) -> None:
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(dim, dim, key=key1)
        self.linear2 = eqx.nn.Linear(dim, dim, key=key2)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run Residual Block layers."""
        residual = x
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        # Residual Net: activated linear F(x) is added to shortcut connection, residual
        return jax.nn.relu(x + residual)


class ResNet(eqx.Module):
    """ResNet model."""

    input_layer: eqx.nn.Linear
    residual_block: ResidualBlock
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        key: jax.Array,
    ) -> None:
        input_key, residual_block_key, output_key = jax.random.split(key, 3)

        # Project input channel up to hidden layer channels
        self.input_layer = eqx.nn.Linear(
            in_features=6, out_features=hidden_dim, key=input_key
        )

        # Run residual block
        self.residual_block = ResidualBlock(dim=hidden_dim, key=residual_block_key)

        # Project hidden channels down to output
        self.output_layer = eqx.nn.Linear(hidden_dim, 2, key=output_key)
        self.output_layer = eqx.tree_at(
            lambda l: l.bias,  # ruff: ignore[ambiguous-variable-name]
            self.output_layer,
            jnp.array([1.0, 0.0]),  # Sets initial hop_time log-scale bias
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Propagate input through model layers."""
        # Normalise inputs
        x = x.at[4].set(
            x[4] * Boltzmann
        )  # Turn temperature into kbt for simplified calculation
        x_mean = jnp.mean(x, axis=0)
        x_std = jnp.std(x, axis=0) + 1e-6
        x = (x - x_mean) / x_std

        # Run model
        x = jax.nn.relu(self.input_layer(x))  # shape = (hidden_dim,)
        x = self.residual_block(x)  # shape = (hidden_dim,)
        x = self.output_layer(x)  # shape = (2,)

        hop_time = jnp.exp(x[0])
        offset = 1.0
        return jnp.array([hop_time, offset])


default_params = KramersParameters(
    omega_well=1.0,
    omega_barrier=1.0,
    barrier_energy=3.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)


@eqx.filter_jit
def loss_fn(
    model: eqx.Module,
    *,
    time_span: TimeSpan,
    delta_k: float,
    test_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function\n")

    test_params, test_hop_times, test_num_hops = test_data
    # Test params: Array of parameters sets in the test data
    # Test hop times: Array of the corresponding hop times derived from mean dwell time in filtered trajectory
    # Test num hops: Array of the corresponding number of hops (i.e. weighting of error) in filtered trajectory 

    # Pass batched params through the model to predict hopping rates and isf offsets
    predictions = jax.vmap(model, (0))(test_params)  # ty: ignore[invalid-argument-type]
    model_hop_times = predictions[:, 0]

    loss = (jnp.log(test_hop_times) - jnp.log(model_hop_times)) * test_num_hops

    # Return the average error
    return jnp.mean(loss)









def _kramers_rate(params: KramersParameters) -> jnp.ndarray:
    return (
        (params.omega_well * params.omega_barrier) / (2 * jnp.pi * params.gamma)
    ) * jnp.exp(-params.barrier_energy / params.kbt)






if __name__ == "__main__":
