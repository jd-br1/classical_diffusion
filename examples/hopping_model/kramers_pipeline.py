import os

import jax.numpy as jnp
import numpy as np
import optax
from scipy.constants import Boltzmann

from classical_diffusion.hopping import (
    CanonicalLattice,
    Lattice1D,
    get_deterministic_isf,
    get_deterministic_probabilities,
)
from classical_diffusion.jax import get_measured_data, get_pairwise_isf
from classical_diffusion.jax.hopping import (
    get_deterministic_isf as get_deterministic_isf_jax,
)
from classical_diffusion.jax.hopping import (
    get_deterministic_probabilities as get_deterministic_probabilities_jax,
)
from classical_diffusion.jax.langevin import (
    KramersParameters,
    KramersSystem1D,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import timed

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import operator
import pickle  # ruff: ignore[suspicious-pickle-import]
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", False)


if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem

CHECK_EVERY = 5
EARLY_STOP = 0.1  # Improvement to loss over CHECK_EVERY epochs deemed small enough to have reached training plateau
NUM_EPOCHS = 100
BATCH_SIZE = 16  # Number of isfs to test on at a time (does this need to be limited?)
LOSS_BATCH_SIZE = 8  # Number of isfs to calculate loss of at a time


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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Propagate input through model layers."""
        x = x.at[4].set(
            x[4] * Boltzmann
        )  # Turn temperature into kbt for simplified calculation
        x = jax.nn.relu(self.input_layer(x))  # shape = (hidden_dim,)
        x = self.residual_block(x)  # shape = (hidden_dim,)
        x = self.output_layer(x)  # shape = (2,)

        hop_time = jnp.exp(x[0]) + 1e-4
        offset = 1.0
        return jnp.array([hop_time, offset])


# Model functions


@jax.checkpoint
@jax.jit
def get_deterministic_isf_directly(
    hop_time: float, time_span: TimeSpan, delta_k: float
) -> jnp.ndarray:
    """Get the isf from hop time directly, without vmapping over get probabilities and get isf."""
    lattice = CanonicalLattice(1.0, hop_time)

    probabilities, _ = get_deterministic_probabilities_jax(
        lattice, time_span, shape=(1000,)
    )

    return get_deterministic_isf_jax(lattice, probabilities, (delta_k,))


@jax.jit
def loss_fn(
    model: eqx.Module,
    *,
    time_span: TimeSpan,
    delta_k: float,
    test_isfs: jnp.ndarray,
    test_params: jnp.ndarray,
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function\n")

    # Pass batched isfs through the model to predict hopping rates and isf offsets
    predictions = jax.vmap(model)(test_params)  # ty: ignore[invalid-argument-type]
    hopping_times = predictions[:, 0]
    offsets = predictions[:, 1]

    total_samples = hopping_times.shape[0]
    num_chunks = total_samples // LOSS_BATCH_SIZE
    usable_samples = num_chunks * LOSS_BATCH_SIZE

    hop_time_chunks = hopping_times[:usable_samples].reshape(
        num_chunks, LOSS_BATCH_SIZE
    )

    # Parallel evaluation inside each batch via vmap
    def _process_loss_batch(hop_time_chunk: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(get_deterministic_isf_directly, (0, None, None))(
            hop_time_chunk, time_span, delta_k
        )

    # Sequential execution over micro-batches via lax.map
    isfs = jax.lax.map(_process_loss_batch, hop_time_chunks)
    isfs = isfs.reshape(usable_samples, -1)

    corrected_isfs = offsets[:usable_samples, None] * isfs
    temp_corrected_isfs = (corrected_isfs * 0.5) + 0.5
    targets = test_isfs[:usable_samples]

    # Calculate the error by
    errors = jnp.sum(
        (temp_corrected_isfs - targets) ** 2,
        axis=-1,
    )

    # Return the average error
    return jnp.mean(errors)


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
    omega_barrier = 1.0
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
    omega_barrier=1.0,
    barrier_energy=1.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)


@timed
def generate_single_clean_isf(
    folderpath: str,
    *,
    time_span: TimeSpan,
    delta_k: float,
    params: KramersParameters = default_params,
) -> None:
    """Run a langevin ensemble and save the resulting clean isf to file."""
    system = KramersSystem1D(params=params).as_canonical()

    times, x_points, _ = solve_many_overdamped_jax(
        system,
        time_span,
        initial_conditions=(jnp.full((500, 1), 0.0), jnp.full((500, 1), 0.0)),
        _key=jax.random.key(33),
    )

    isf = get_pairwise_isf(x_points, jnp.asarray((delta_k,)))
    avg_isf = jnp.mean(isf, axis=0)
    sem_isf = jnp.std(isf, axis=0) / np.sqrt(isf.shape[0])

    avg_data = get_measured_data(avg_isf, "real")
    sem_data = get_measured_data(sem_isf, "real")

    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"
    with Path(clean_isf_path).open("wb") as file:
        isf_record = {"time": times, "isf": avg_data, "error": sem_data}
        pickle.dump(isf_record, file)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line,) = ax.plot(times, avg_data)
    line.set_label("ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Clean Langevin ISF")

    fig.savefig("./examples/hopping_model/test.clean_isf.pdf")


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


@timed
def generate_isfs(
    traj_filepath: str, isfs_filepath: str, traj_per_isf: int, *, delta_k: float
) -> None:
    """Generate isfs from trajectories saved in ML pipeline and save to file."""
    # Open the trajectories file and load in the trajectories
    with Path(traj_filepath).open("rb") as file:
        batched = pickle.load(file)

    times_all, x_points_all = batched["result"]
    parameters_all = batched["parameters"]
    n_traj = x_points_all.shape[0]

    # Open the isfs file and calculate, then save, the isfs
    with Path(isfs_filepath).open("wb") as file:
        for i in range(0, n_traj, traj_per_isf):
            x_chunk = x_points_all[i : i + traj_per_isf]
            times = times_all[i]
            params = jax.tree.map(operator.itemgetter(i), parameters_all)

            isf = get_pairwise_isf(jnp.array(x_chunk), jnp.asarray((delta_k,)))
            avg_isf = jnp.mean(isf, axis=0)
            sem_isf = jnp.std(isf, axis=0) / np.sqrt(isf.shape[0])

            avg_data = get_measured_data(avg_isf, "real")[0]
            sem_data = get_measured_data(sem_isf, "real")[0]

            isf_record = {
                "parameters": params,
                "isf": {"time": times, "isf": avg_data, "error": sem_data},
            }
            pickle.dump(isf_record, file)


# Train & Test functions


def untrained_test(folderpath: str) -> None:
    """Test an untrained model with a clean isf input."""
    print("\n\nRunning untrained test\n")

    key = jax.random.key(40)
    model = ResNet(key=key)

    with Path(folderpath + "/langevin_clean_isf.pkl").open("rb") as file:
        isf_record = pickle.load(file)
    x_input = jnp.array([isf_record.get("isf")])
    output = model(x_input)

    print("Input shape:", x_input.shape)
    print("Output shape:", output.shape)
    print("Output value:", output)


# Broken
def single_clean_test(folderpath: str) -> None:
    """Train and test a model on a single, clean ISF."""
    print("\n\nRunning test with a single, clean isf\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    delta_k = 0.5

    # Define filepath
    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"

    # Ensure isf exists
    if not Path(clean_isf_path).exists():
        print("No data, generating a new clean isf")
        generate_single_clean_isf(folderpath, time_span=time_span, delta_k=delta_k)

    # Load isf
    with Path(clean_isf_path).open("rb") as file:
        clean_isf = pickle.load(file)

    # Train model on clean isf
    training_isf = jnp.array(clean_isf.get("isf"))
    training_error = jnp.array(clean_isf.get("error"))

    print("Training model")
    trained_model = train_model(
        (training_isf[None, None, :], training_error[None, None, :]),
        time_span=time_span,
        delta_k=delta_k,
    )

    print("\nModel trained! Getting model's isf")

    # Test model on clean isf
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, training_isf[None, :]
    )
    hopping_time = float(hopping_time)
    offset = float(offset)

    predicted_lattice = Lattice1D(1.0, hopping_time)
    model_isf = get_deterministic_isf(
        get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
        (delta_k,),
    )

    corrected_model_isf = offset * model_isf

    # Plot source and model isfs for comparison
    print("Plotting ISFs")
    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(clean_isf.get("time"), clean_isf.get("isf"))
    line1.set_label("Langevin ISF")

    (line2,) = ax.plot(clean_isf.get("time"), corrected_model_isf)
    line2.set_label("Model ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Single clean ISF comparison")

    fig.savefig("./examples/hopping_model/model_test_single_clean.isf.pdf")


def many_test(folderpath: str, n_isfs: int, resume: bool = False) -> None:  # ruff: ignore[too-many-locals, too-many-statements]
    """Train and test a model on n_isfs equivalent but noisy ISFs."""
    print(f"\n\nRunning test with {n_isfs} equivalent isfs\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    delta_k = 0.5
    traj_per_isf = 10

    # Define filepaths
    traj_filepath = folderpath + f"/langevin_{n_isfs * traj_per_isf}_equivalent.pkl"
    isfs_filepath = (
        folderpath
        + f"/langevin_{n_isfs}_equivalent_isf_averaged_over_{traj_per_isf}.pkl"
    )

    # Ensure isfs exist
    if not Path(traj_filepath).exists():
        print("No data, generating new equivalent trajectories")
        generate_many_langevin_trajectories(
            traj_filepath, n_isfs * traj_per_isf, time_span=time_span
        )
    if not Path(isfs_filepath).exists():
        print("No isfs, generating new equivalent isfs")
        generate_isfs(traj_filepath, isfs_filepath, traj_per_isf, delta_k=delta_k)

    # Load isfs
    training_isf_data = []
    with Path(isfs_filepath).open("rb") as file:
        while True:
            try:
                training_isf_data.append(pickle.load(file))  # ruff: ignore[suspicious-pickle-usage]
            except EOFError:
                break

    # Train model: select training data
    training_isfs = jnp.array(
        [data.get("isf").get("isf") for data in training_isf_data]
    )
    training_errors = jnp.array(
        [data.get("isf").get("error") for data in training_isf_data]
    )

    # Train model: train model
    print("Training model")
    trained_model = train_model(
        (training_isfs[:, None, :], training_errors[:, None, :]),
        time_span=time_span,
        delta_k=delta_k,
        resume=resume,
    )

    print("\nModel trained! Getting model's isf")

    # Test model: get test data
    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"

    # Ensure isf exists
    if not Path(clean_isf_path).exists():
        print("No data, generating a new clean isf")
        generate_single_clean_isf(folderpath, time_span=time_span, delta_k=delta_k)

    # Load isf
    with Path(clean_isf_path).open("rb") as file:
        test_isf = pickle.load(file)

    # Test model: get model output
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, jnp.array([test_isf.get("isf")])
    )
    hopping_time = float(hopping_time)
    offset = float(offset)

    predicted_lattice = Lattice1D(1.0, hopping_time)
    model_isf = get_deterministic_isf(
        get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
        (delta_k,),
    )

    corrected_model_isf = offset * model_isf

    print("Plotting ISFs")
    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(test_isf.get("time"), test_isf.get("isf"))
    line1.set_label("Langevin ISF")

    (line2,) = ax.plot(test_isf.get("time"), corrected_model_isf)
    line2.set_label("Model ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model trained on many, tested on clean")

    fig.savefig("./examples/hopping_model/model_test_many_equiv.isf.pdf")

    i = 0
    for data in training_isf_data:
        isf = data.get("isf")
        hopping_time, offset = get_hopping_time_and_offset(
            trained_model, jnp.array([isf.get("isf")])
        )
        hopping_time = float(hopping_time)
        offset = float(offset)

        predicted_lattice = Lattice1D(1.0, hopping_time)
        model_isf = get_deterministic_isf(
            get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
            (delta_k,),
        )

        corrected_model_isf = offset * model_isf

        fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)
        (line1,) = ax.plot(isf.get("time"), isf.get("isf"))
        line1.set_label("Langevin ISF")

        (line2,) = ax.plot(isf.get("time"), corrected_model_isf)
        line2.set_label("Model ISF")

        fill = ax.fill_between(
            isf.get("time"),
            isf.get("isf") - isf.get("error"),
            isf.get("isf") + isf.get("error"),
        )
        fill.set_alpha(0.3)
        fill.set_color(line1.get_color())

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=40)
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.set_title("Model trained on many, tested on random")

        i += 1
        fig.savefig(f"./examples/hopping_model/training_isfs/test_{i}.isf.pdf")

        if i == 15:
            break


def kramers_rate_plot_test(folderpath: str, resume: bool = False) -> None:
    """Train a model on a range of omega_well values and correlate ."""
    print("\n\nRunning kamers rate theory\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    delta_k = 0.5

    # Test parameters
    traj_per_isf = 100
    n_parameter_data_points = 50
    # 9 training isfs per test isf
    n_isfs = 10 * n_parameter_data_points

    # Define filepaths
    traj_filepath = (
        folderpath
        + f"/vary_omega_well/langevin_{n_isfs * traj_per_isf}_trajectories.pkl"
    )
    Path(traj_filepath).parent.mkdir(parents=True, exist_ok=True)
    isfs_filepath = (
        folderpath
        + f"/vary_omega_well/langevin_{n_isfs}_isf_averaged_over_{traj_per_isf}.pkl"
    )

    # Ensure isfs exist
    if not Path(traj_filepath).exists():
        print(f"No data, generating {n_isfs * traj_per_isf} equivalent trajectories")
        generate_many_langevin_trajectories(
            traj_filepath,
            n_isfs * traj_per_isf,
            time_span=time_span,
            generate_params=_vary_omega_well,
        )
    if not Path(isfs_filepath).exists():
        print("No isfs, generating new equivalent isfs")
        generate_isfs(traj_filepath, isfs_filepath, traj_per_isf, delta_k=delta_k)

    # Load isfs
    generated_isf_data = []
    with Path(isfs_filepath).open("rb") as file:
        while True:
            try:
                generated_isf_data.append(pickle.load(file))  # ruff: ignore[suspicious-pickle-usage]
            except EOFError:
                break

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
            i
            for i, in_test in zip(generated_isf_data, is_test, strict=True)
            if not in_test
        ]

        return test_list, training_list

    print("\nSplitting isfs into training and evenly spaced test set")
    test_list, training_list = split_isfs_by_omega_well(
        generated_isf_data, n_parameter_data_points
    )

    # Train model: select training data
    training_isfs = jnp.array([data["isf"]["isf"] for data in training_list])
    training_params = jnp.array([data["parameters"] for data in training_list])

    print("\nTraining model")
    trained_model = train_model(
        (training_isfs, training_params),
        time_span=time_span,
        delta_k=delta_k,
        resume=resume,
    )

    print("\nModel trained! Testing on test data")
    test_params = jnp.array([data["parameters"] for data in test_list])

    # Test model: get model output
    hopping_times, _ = jax.vmap(get_hopping_time_and_offset, (None, 0))(
        trained_model, test_params
    )
    hopping_rate = 1.0 / hopping_times
    omega_wells = jnp.array([data["parameters"][0] for data in test_list])

    i = 0
    times = test_list[0]["isf"]["time"]
    for i in range(len(test_list)):
        isf = test_list[i]["isf"]["isf"]
        hopping_time = hopping_times[i]
        predicted_lattice = Lattice1D(1.0, float(hopping_time))
        model_isf = get_deterministic_isf(
            get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
            (delta_k,),
        )
        temp_isf = (model_isf * 0.5) + 0.5

        fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)
        (line1,) = ax.plot(times, isf)
        line1.set_label("Langevin ISF")

        (line2,) = ax.plot(times, temp_isf)
        line2.set_label("Model ISF")

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=40)
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.set_title("Kramers model test")

        i += 1
        fig.savefig(f"./examples/hopping_model/training_isfs/kramers_test_{i}.isf.pdf")

        if i == 15:
            break

    # Plot hopping rate against omega_well
    print("Plot Kramers stuff")
    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(omega_wells, hopping_rate)
    line1.set_label("Rate against omega_well")

    ax.set_xlabel("Omega well")
    ax.set_ylabel("Hopping rate")

    ax.set_xlim(0, right=1)
    ax.set_ylim(0, 0.2)
    ax.legend()
    ax.set_title("Kramers")

    fig.savefig("./examples/hopping_model/kramers.linear.pdf")

    print(hopping_times)


def pre_generate_on_gpu(folderpath: str) -> None:
    """Pre-generate training isfs."""
    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)

    num_trajs = [10000, 100000, 1000000]

    for num_traj in num_trajs:
        traj_filepath = (
            folderpath + f"/vary_barrier_energy/langevin_{num_traj}_equivalent.pkl"
        )
        Path(traj_filepath).parent.mkdir(parents=True, exist_ok=True)

        if not Path(traj_filepath).exists():
            print(f"Generating {num_traj} trajectories")
            generate_many_langevin_trajectories(
                traj_filepath, num_traj, time_span=time_span
            )


if __name__ == "__main__":
    path = "./examples/data"
    # many_test(path, 20)
    # pre_generate_on_gpu(path)
    kramers_rate_plot_test(path)
