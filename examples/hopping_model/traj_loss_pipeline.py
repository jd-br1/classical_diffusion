import itertools
import os

import jax.numpy as jnp
import numpy as np
import optax
from scipy.constants import Boltzmann

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

jax.config.update("jax_enable_x64", False)


if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem
    from classical_diffusion.simulation import TimeSpan

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
            lambda l: l.bias,
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


# Model functions


def get_hopping_time_and_offset(
    model: ResNet, params: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the hopping time and initial offset from a pre-trained model."""
    outputs = model(params)
    return outputs[0], outputs[1]


# Training functions


def train_model(
    training_data: tuple[jnp.ndarray, jnp.ndarray],
    *,
    time_span: TimeSpan,
    delta_k: float,
    resume: bool = False,
) -> ResNet:
    """Train a ResNet model."""
    # Set up constants
    key = jax.random.key(1)

    fresh_model = ResNet(hidden_dim=16, key=key)
    if resume:
        model = eqx.tree_deserialise_leaves("model_checkpoint.eqx", fresh_model)
    else:
        model = fresh_model

    optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    training_isfs, training_params = training_data

    # Define number batches per epoch
    num_isfs = len(training_isfs)
    num_batches = num_isfs // BATCH_SIZE
    usable_len = num_batches * BATCH_SIZE

    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def batch_step(
        carry: tuple[eqx.Module, optax.OptState], batch: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[eqx.Module, optax.OptState], jnp.ndarray]:
        model, opt_state = carry
        batch_isfs, batch_params = batch

        batch_loss, gradients = loss_and_grad_fn(
            model,
            time_span=time_span,
            delta_k=delta_k,
            test_isfs=batch_isfs,
            test_params=batch_params,
        )

        updates, opt_state = optimizer.update(gradients, opt_state)

        model = eqx.apply_updates(model, updates)

        return (model, opt_state), batch_loss

    @eqx.filter_jit
    def run_epoch(
        model: eqx.Module,
        opt_state: optax.OptState,
        isfs: jnp.ndarray,
        params: jnp.ndarray,
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        """Run one epoch of ML algorithm."""
        batched_isfs = isfs[:usable_len].reshape(
            num_batches, BATCH_SIZE, *isfs.shape[1:]
        )
        batched_params = params[:usable_len].reshape(
            num_batches, BATCH_SIZE, *params.shape[1:]
        )

        (model, opt_state), batch_losses = jax.lax.scan(
            batch_step, (model, opt_state), (batched_isfs, batched_params)
        )

        return model, opt_state, jnp.sum(batch_losses)

    # Control loop outside jit
    losses = []
    for epoch in range(NUM_EPOCHS):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_isfs)

        shuffled_isfs = training_isfs[permutation]
        shuffled_params = training_params[permutation]

        model, optimizer_state, epoch_loss = run_epoch(
            model, optimizer_state, shuffled_isfs, shuffled_params
        )
        losses.append(float(epoch_loss))
        print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss:.5f}")

        if (epoch + 1) % CHECK_EVERY == 0:
            eqx.tree_serialise_leaves("model_checkpoint.eqx", model)
            loss_difference = losses[-CHECK_EVERY] - epoch_loss
            if (
                len(losses) >= CHECK_EVERY
                and loss_difference >= 0
                and loss_difference < EARLY_STOP
            ):
                print("Early stopping triggered: plateau reached.")
                break

    # Return trained model
    return model  # ty: ignore[invalid-return-type]


# GENERATE TRAJECTORIES
# Simulation input generator functions
@jax.jit
def _vary_barrier_energy(
    _key: jax.Array,
) -> tuple[tuple[float, float, float, float, float, float], "CanonicalSystem"]:  # ruff: ignore[quoted-annotation]

    rand = jax.random.uniform(_key, shape=(), minval=0.5, maxval=3.0)
    omega_well = 1.0
    omega_barrier = 1.0
    barrier_energy = rand.astype(jnp.float32)
    m = 1.0
    temperature = 0.5 / Boltzmann
    gamma = 0.1

    params = (omega_well, omega_barrier, barrier_energy, m, temperature, gamma)
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=omega_well,
            omega_barrier=omega_barrier,
            barrier_energy=barrier_energy,  # ty: ignore[invalid-argument-type]
            m=m,
            temperature=temperature,
            gamma=gamma,
        )
    ).as_canonical()

    return params, system  # ty: ignore[invalid-return-type]


@jax.jit
def _vary_omega_well(
    _key: jax.Array,
) -> tuple[tuple[float, float, float, float, float, float], "CanonicalSystem"]:  # ruff: ignore[quoted-annotation]

    rand = jax.random.uniform(_key, shape=(), minval=0.25, maxval=1.0)
    omega_well = rand.astype(jnp.float32)
    omega_barrier = 10.0
    barrier_energy = 3.0
    m = 1.0
    temperature = 0.5 / Boltzmann
    gamma = 0.1

    params = (omega_well, omega_barrier, barrier_energy, m, temperature, gamma)
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=omega_well,  # ty: ignore[invalid-argument-type]
            omega_barrier=omega_barrier,
            barrier_energy=barrier_energy,
            m=m,
            temperature=temperature,
            gamma=gamma,
        )
    ).as_canonical()

    return params, system  # ty: ignore[invalid-return-type]


@jax.jit
def _constant(
    _key: jax.Array,
) -> tuple[tuple[float, float, float, float, float, float], "CanonicalSystem"]:  # ruff: ignore[quoted-annotation]

    omega_well = 1.0
    omega_barrier = 10.0
    barrier_energy = 3.0
    m = 1.0
    temperature = 0.5 / Boltzmann
    gamma = 0.1

    params = (omega_well, omega_barrier, barrier_energy, m, temperature, gamma)
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=omega_well,  # ty: ignore[invalid-argument-type]
            omega_barrier=omega_barrier,
            barrier_energy=barrier_energy,
            m=m,
            temperature=temperature,
            gamma=gamma,
        )
    ).as_canonical()

    return params, system  # ty: ignore[invalid-return-type]


@jax.jit
def _inits_constant(
    _key: jax.Array,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
]:
    const_initial_cond = jnp.full((1, 1), 0.0)
    return (const_initial_cond, const_initial_cond)


# jax simulation run
@eqx.filter_jit
def run_langevin_trajectories(
    *,
    time_span: TimeSpan,
    keys: jax.Array,
    generate_params: "Callable[        [jax.Array],        tuple[tuple[float, float, float, float, float, float], CanonicalSystem],    ]",  # ruff: ignore[quoted-annotation]
) -> JaxEnsembleResults:
    """Run langevin trajectories."""

    def body(time_span: TimeSpan, _key: jax.Array) -> JaxEnsembleResults:

        param_key, cond_key, sim_key = jax.random.split(_key, 3)

        params, system = generate_params(param_key)

        initial_conditions = _inits_constant(cond_key)

        times, positions, _ = solve_many_overdamped_jax(
            system,
            time_span,
            initial_conditions,
            _key=sim_key,
        )

        return {
            "parameters": params,  # ty: ignore[invalid-argument-type]
            "initial_conditions": initial_conditions,
            "result": (times, positions),
        }  # ty: ignore[invalid-return-type]

    return jax.vmap(body, (None, 0))(time_span, keys)


# Training data generation functions


default_params = KramersParameters(
    omega_well=1.0,
    omega_barrier=10.0,
    barrier_energy=3.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)


@timed
def generate_many_langevin_trajectories(
    traj_filepath: str,
    n_traj: int,
    *,
    time_span: TimeSpan,
    generate_params: "Callable[        [jax.Array],        tuple[tuple[float, float, float, float, float, float], CanonicalSystem],    ]" = _vary_barrier_energy,  # ruff: ignore[quoted-annotation]
) -> None:
    """Run the langevin simulations and save to file."""
    keys = jax.random.split(jax.random.key(100), n_traj)

    # Transfer batched array directly from GPU to CPU as contiguous blocks
    batched_trajectories = jax.tree.map(
        jnp.asarray,
        run_langevin_trajectories(
            time_span=time_span, keys=keys, generate_params=generate_params
        ),
    )

    # Save directly without slicing into 10,000 Python objects
    with Path(traj_filepath).open("wb") as file:
        pickle.dump(batched_trajectories, file, protocol=pickle.HIGHEST_PROTOCOL)


# FILTER TRAJECTORIES

MIN_SPLIT_POINTS = 4
MIN_VARIANCE = 1e-15


@jax.jit
def filter_trajectory_jax(
    x: jnp.ndarray,
    *,
    process_points: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Discretize the signal using the objective Kalafut-Visscher step detection algorithm."""  # cspell: disable-line
    if process_points is None:

        def process_points(u):
            return u

    N = x.shape[0]

    # Precalculate prefix sums for O(1) segment variance and sum-of-squares evaluation
    p1 = jnp.pad(jnp.cumsum(x), (1, 0))
    p2 = jnp.pad(jnp.cumsum(x**2), (1, 0))

    # Initialize fixed-size execution stack and boundary tracking
    stack_starts = jnp.zeros(N, dtype=jnp.int32).at[0].set(0)
    stack_ends = jnp.zeros(N, dtype=jnp.int32).at[0].set(N)
    stack_ptr = jnp.int32(1)
    breakpoints = jnp.zeros(N + 1, dtype=jnp.bool_).at[0].set(True).at[N].set(True)

    init_state = (stack_starts, stack_ends, stack_ptr, breakpoints)

    def cond_fn(
        state: tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray],
    ) -> jax.Array:
        _, _, ptr, _ = state
        return ptr > 0

    def body_fn(  # ruff: ignore[too-many-locals]
        state: tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray]:
        starts, ends, ptr, bpts = state

        # Pop current top segment
        pop_ptr = ptr - 1
        start = starts[pop_ptr]
        end = ends[pop_ptr]
        n_seg = end - start

        # Base segment statistics
        s1 = p1[end] - p1[start]
        s2 = p2[end] - p2[start]
        base_mu = process_points(s1 / jnp.maximum(n_seg, 1))
        base_ssr = s2 - 2 * base_mu * s1 + n_seg * base_mu**2
        base_var = base_ssr / jnp.maximum(n_seg, 1)

        can_split = (n_seg > MIN_SPLIT_POINTS) & (base_var > MIN_VARIANCE)

        # Evaluate all candidate split points k in parallel across length N+1
        k = jnp.arange(N + 1)
        k_left_len = k - start
        k_right_len = end - k

        s1_left = p1[k] - p1[start]
        s1_right = s1 - s1_left

        mu_left = process_points(s1_left / jnp.maximum(k_left_len, 1))
        mu_right = process_points(s1_right / jnp.maximum(k_right_len, 1))

        split_ssr = (
            s2
            - 2 * mu_left * s1_left
            + k_left_len * mu_left**2
            - 2 * mu_right * s1_right
            + k_right_len * mu_right**2
        )

        split_vars = jnp.maximum(split_ssr / jnp.maximum(n_seg, 1), MIN_VARIANCE)
        delta_sic = n_seg * jnp.log(base_var / split_vars) - jnp.log(
            jnp.maximum(n_seg, 1)
        )

        # Mask invalid candidate split points: start + 2 <= k <= end - 2
        valid_k_mask = (k >= start + 2) & (k <= end - 2)
        delta_sic_masked = jnp.where(valid_k_mask, delta_sic, -jnp.inf)

        best_k = jnp.argmax(delta_sic_masked)
        best_delta = delta_sic_masked[best_k]

        should_split = can_split & (best_delta > 0)

        # Update breakpoints and stack buffers
        new_bpts = jnp.where(should_split, bpts.at[best_k].set(True), bpts)

        new_starts = jnp.where(
            should_split,
            starts.at[pop_ptr].set(start).at[pop_ptr + 1].set(best_k),
            starts,
        )
        new_ends = jnp.where(
            should_split,
            ends.at[pop_ptr].set(best_k).at[pop_ptr + 1].set(end),
            ends,
        )
        new_ptr = jnp.where(should_split, pop_ptr + 2, pop_ptr)

        return (new_starts, new_ends, new_ptr, new_bpts)

    # Execute bounded segmentation loop
    _, _, _, final_breakpoints = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # Reconstruct piecewise constant path in parallel
    segment_ids = jnp.cumsum(final_breakpoints[:-1]) - 1
    counts = jnp.bincount(segment_ids, length=N)
    sums = jnp.bincount(segment_ids, weights=x, length=N)

    segment_means = jnp.where(counts > 0, sums / jnp.maximum(counts, 1), 0.0)
    processed_means = process_points(segment_means)

    return processed_means[segment_ids]


@timed
def filter_trajectory(  # ruff: ignore[too-many-locals]
    x: np.ndarray[Any, np.dtype[np.float64]],
    *,
    process_points: Callable[
        [np.ndarray[Any, np.dtype[np.floating]] | np.floating],
        np.ndarray[Any, np.dtype[np.floating]] | np.floating,
    ]
    | None = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Discretize the signal using the objective Kalafut-Visscher step detection algorithm."""  # cspell: disable-line
    process_points = process_points or (lambda x: x)
    breakpoints = [0, len(x)]
    stack = [(0, len(x))]

    while stack:
        start, end = stack.pop()
        n_seg = end - start

        if n_seg <= MIN_SPLIT_POINTS:
            continue

        segment = x[start:end]
        total_sum = np.sum(segment)
        total_sum_x2 = np.sum(segment**2)

        # Calculate local baseline variance before splitting
        base_mu = process_points(total_sum / n_seg)
        base_ssr = total_sum_x2 - 2 * base_mu * total_sum + n_seg * base_mu**2
        base_variance = base_ssr / n_seg
        if base_variance <= MIN_VARIANCE:
            continue

        indices = np.arange(2, n_seg - 1)
        n_right_arr = n_seg - indices

        cum_sum_left = np.cumsum(segment)[indices - 1]
        cum_sum_right = total_sum - cum_sum_left

        mu_left = process_points(cum_sum_left / indices)
        mu_right = process_points(cum_sum_right / n_right_arr)
        # Vectorized reduction in sum of squared residuals
        split_ssr = (
            total_sum_x2
            - 2 * mu_left * cum_sum_left
            + indices * mu_left**2
            - 2 * mu_right * cum_sum_right
            + n_right_arr * mu_right**2
        )
        split_variances = split_ssr / n_seg
        split_variances = np.maximum(split_variances, MIN_VARIANCE)

        # Objective Criterion: delta_sic > 0 means the split is justified by the data
        delta_sic = n_seg * np.log(base_variance / split_variances) - np.log(n_seg)
        best_idx = np.argmax(delta_sic)

        if delta_sic[best_idx] > 0:
            global_split_point = start + indices[best_idx]
            breakpoints.append(global_split_point)
            stack.extend([(start, global_split_point), (global_split_point, end)])

    # Reconstruct the final denoised piece-wise constant path
    breakpoints = sorted(set(breakpoints))
    fitted_trajectory = np.zeros_like(x)

    for b_start, b_end in itertools.pairwise(breakpoints):
        fitted_trajectory[b_start:b_end] = process_points(np.mean(x[b_start:b_end]))

    return fitted_trajectory


# Compare two algorithms

if __name__ == "__main__":
    traj_path = "./examples/data/vary_barrier_energy/langevin_8000_trajectories.pkl"
    with Path(traj_path).open("rb") as file:
        batched = pickle.load(file)

    times_all, x_points_all = batched["result"]
    times = np.array(times_all[0])
    x_points = np.array(x_points_all[0])

    print(times.shape)
    print(x_points.shape)

    filtered_trajectory = filter_trajectory(x_points)

    filtered_trajectory_jax = filter_trajectory_jax(jnp.array(x_points))

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(times, x_points)
    line1.set_label("Trajectory")

    (line1,) = ax.plot(times, filtered_trajectory)
    line1.set_label("Filtered Trajectory")

    (line1,) = ax.plot(times, filtered_trajectory_jax)
    line1.set_label("Jax's Filtered Trajectory")

    ax.set_xlabel("Omega well")
    ax.set_ylabel("Hopping rate")

    ax.set_xlim(0, right=1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Kramers")

    fig.savefig("./examples/hopping_model/kramers.linear.pdf")


"""
def _combine_close_jumps(
    x: np.ndarray[Any, np.dtype[np.float64]],
    window: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    diff = np.diff(x, prepend=x[0])
    non_zero = np.argwhere(diff != 0)[:, 0]
    for i in range(len(non_zero) - 1):
        arg = non_zero[i]
        next_arg = non_zero[i + 1]
        if (next_arg - arg) < window and np.sign(diff[arg]) == np.sign(diff[next_arg]):
            old_lhs = np.abs(diff[arg])
            old_rhs = np.abs(diff[next_arg])
            new_rhs_arg = int(
                (old_lhs * arg + old_rhs * next_arg) / (old_lhs + old_rhs)
            )
            new_rhs_val = diff[arg] + diff[next_arg]
            diff[next_arg] = 0
            diff[arg] = 0
            diff[new_rhs_arg] = new_rhs_val
            non_zero[i + 1] = new_rhs_arg
            continue
    return np.cumsum(diff)


def _prune_small_jumps(
    x: np.ndarray[Any, np.dtype[np.float64]],
    min_jump: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    diff = np.diff(x, prepend=x[0])
    non_zero = np.argwhere(diff != 0)[:, 0]
    for i in range(len(non_zero) - 1):
        arg = non_zero[i]
        next_arg = non_zero[i + 1]
        if np.abs(diff[arg]) < min_jump:
            diff[next_arg] += diff[arg]
            diff[arg] = 0
            continue
    return np.cumsum(diff)


def get_discretized_trajectory(
    result: AlphaSimulationResult[PeriodicParameters], *, simplify: bool = True
) -> AlphaSimulationResult[PeriodicParameters]:
    x = result.x[0]
    delta_x = result.params.as_si().delta_x
    scaled_x = (x / delta_x) - 0.5
    dt = result.times[1] - result.times[0]
    window = int(1 / (dt * result.params.as_si().lambda_))
    # Use Kalafut-Visscher step detection algorithm to simplify the trajectory # cspell: disable-line
    denoised_x = filter_trajectory_kv(scaled_x)
    if simplify:
        # We must combine before puning, otherwise successive small jumps are incorrectly pruned
        denoised_x = _combine_close_jumps(denoised_x, window=window)
        denoised_x = _prune_small_jumps(denoised_x, min_jump=0.5)
        # Second pass combines large jumps that were seperated by a small backwards jump
        denoised_x = _combine_close_jumps(denoised_x, window=window)
        # Place average at the nearest lattice point
        denoised_x = np.round(denoised_x)
    denoised_x = (denoised_x.reshape(1, -1) + 0.5) * delta_x
    return AlphaSimulationResult(
        params=result.params,
        simulation_times=result.simulation_times,
        alpha=denoised_x / (np.sqrt(2) * result.params.as_si().lengthscale)
        + 1j * result.alpha.imag,
    )
"""  # ruff: ignore[too-many-blank-lines]


@timed
def split_isfs_by_omega_well(
    generated_isf_data: list[dict], n: int
) -> tuple[list[dict], list[dict]]:
    total_len = len(generated_isf_data)

    # Sort isfs by parameter of interest: omega_well
    omega_well_values = jnp.array(
        [data["parameters"][0] for data in generated_isf_data]
    )

    sorted_indices = jnp.argsort(omega_well_values)
    sorted_omega = omega_well_values[sorted_indices]

    targets = jnp.linspace(sorted_omega[0], sorted_omega[-1], n)
    target_positions = jnp.searchsorted(sorted_omega, targets)
    target_positions = jnp.clip(target_positions, 0, total_len - 1)

    selected_indices = np.asarray(sorted_indices[target_positions])

    is_test = np.zeros(total_len, dtype=bool)
    is_test[selected_indices] = True

    test_list = [
        i for i, in_test in zip(generated_isf_data, is_test, strict=True) if in_test
    ]
    training_list = [
        i for i, in_test in zip(generated_isf_data, is_test, strict=True) if not in_test
    ]

    return test_list, training_list
