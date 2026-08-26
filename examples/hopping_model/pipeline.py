import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import operator
import pickle  # ruff: ignore[suspicious-pickle-import]
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import equinox as eqx
import jax
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
from classical_diffusion.langevin import solve_many_overdamped
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem

CHECK_EVERY = 2
EARLY_STOP = 1  # Improvement to loss over CHECK_EVERY epochs deemed small enough to have reached training plateau
NUM_EPOCHS = 100
BATCH_SIZE = 1


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

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, channels: int, kernel_size: int = 8, *, key: jax.Array) -> None:
        key1, key2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv1d(
            channels, channels, kernel_size, padding="same", key=key1
        )
        self.conv2 = eqx.nn.Conv1d(
            channels, channels, kernel_size, padding="same", key=key2
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run Residual Block layers."""
        residual = x
        x = jax.nn.relu(self.conv1(x))
        x = self.conv2(x)
        # Residual Net: activated linear F(x) is added to shortcut connection, residual
        return jax.nn.relu(x + residual)


class ResNet(eqx.Module):
    """ResNet model."""

    input_layer: eqx.nn.Conv1d
    residual_block: ResidualBlock
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        key: jax.Array,
    ) -> None:
        input_key, residual_block_key, output_key = jax.random.split(key, 3)

        # Project input channel up to hidden layer channels
        self.input_layer = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_channels, kernel_size=8, key=input_key
        )

        # Run residual block
        self.residual_block = ResidualBlock(
            channels=hidden_channels, key=residual_block_key
        )

        # Project hidden channels down to output
        self.output_layer = eqx.nn.Linear(hidden_channels, 2, key=output_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Propagate input through model layers."""
        x = jax.nn.relu(self.input_layer(x))  # shape = (hidden_channels, n_time_steps)
        x = self.residual_block(x)  # shape = (hidden_channels, n_time_steps)

        x = jnp.mean(x, axis=-1)  # shape = (hidden_channels,)
        x = self.output_layer(x)  # shape = (2,)

        return x.at[1].set(1)  # 0.5 + 0.5 * jax.nn.tanh(x[1]))


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
    test_errors: jnp.ndarray,
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function\n")
    # Pass batched isfs through the model to predict hopping rates and isf offsets
    predictions = jax.vmap(model)(test_isfs)  # ty: ignore[invalid-argument-type]

    # For each prediction, generate an isf
    hopping_times = predictions[:, 0]
    offsets = predictions[:, 1]

    isfs = jax.vmap(get_deterministic_isf_directly, (0, None, None))(
        hopping_times, time_span, delta_k
    )  # These are already real

    corrected_isfs = offsets[:, None] * isfs

    # For each isf, compare to test isf
    # eps = 1e-8
    # chi_squared_errors = jnp.sum(
    #     ((corrected_isfs - test_isfs.squeeze(axis=1)) / (eps * test_errors.squeeze(axis=1)))
    #     ** 2,
    #     axis=-1,
    # )
    errors = jnp.sum(
        (corrected_isfs - test_isfs.squeeze(axis=1)) ** 2,
        axis=-1,
    )

    # Return the average error
    return jnp.mean(errors)


def get_hopping_time_and_offset(model: ResNet, isf: jnp.ndarray) -> tuple[float, float]:
    """Get the hopping time and initial offset from a pre-trained model."""
    outputs = model(isf)
    return float(outputs[0]), float(outputs[1])


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

    fresh_model = ResNet(hidden_channels=16, key=key)
    if resume:
        model = eqx.tree_deserialise_leaves("model_checkpoint.eqx", fresh_model)
    else:
        model = fresh_model

    optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    training_isfs, training_errors = training_data

    # Define number batches per epoch
    num_isfs = len(training_isfs)
    num_batches = num_isfs // BATCH_SIZE
    usable_len = num_batches * BATCH_SIZE

    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def batch_step(
        carry: tuple[eqx.Module, optax.OptState], batch: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[eqx.Module, optax.OptState], jnp.ndarray]:
        model, opt_state = carry
        batch_isfs, batch_errors = batch

        batch_loss, gradients = loss_and_grad_fn(
            model,
            time_span=time_span,
            delta_k=delta_k,
            test_isfs=batch_isfs,
            test_errors=batch_errors,
        )

        updates, opt_state = optimizer.update(gradients, opt_state)

        model = eqx.apply_updates(model, updates)

        return (model, opt_state), batch_loss

    @eqx.filter_jit
    def run_epoch(
        model: eqx.Module,
        opt_state: optax.OptState,
        isfs: jnp.ndarray,
        errors: jnp.ndarray,
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        """Run one epoch of ML algorithm."""
        batched_isfs = isfs[:usable_len].reshape(
            num_batches, BATCH_SIZE, *isfs.shape[1:]
        )
        batched_errors = errors[:usable_len].reshape(
            num_batches, BATCH_SIZE, *errors.shape[1:]
        )

        (model, opt_state), batch_losses = jax.lax.scan(
            batch_step, (model, opt_state), (batched_isfs, batched_errors)
        )

        return model, opt_state, jnp.sum(batch_losses)

    # Control loop outside jit
    losses = []
    for epoch in range(NUM_EPOCHS):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_isfs)

        shuffled_isfs = training_isfs[permutation]
        shuffled_errors = training_errors[permutation]

        model, optimizer_state, epoch_loss = run_epoch(
            model, optimizer_state, shuffled_isfs, shuffled_errors
        )
        losses.append(float(epoch_loss))

        if (epoch + 1) % CHECK_EVERY == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss:.5f}")
            eqx.tree_serialise_leaves("model_checkpoint.eqx", model)
            print(len(losses))
            if len(losses) >= CHECK_EVERY and (
                losses[-CHECK_EVERY] - epoch_loss < EARLY_STOP
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

    rand = jax.random.uniform(_key, shape=(), minval=0.25, maxval=3.25)
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
@jax.jit
def run_langevin_trajectories(
    *,
    time_span: TimeSpan,
    keys: jax.Array,
    generate_params: "Callable[        [jax.Array],        tuple[tuple[float, float, float, float, float, float], CanonicalSystem],    ]" = _vary_barrier_energy,  # ruff: ignore[quoted-annotation]
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


@timed
def generate_single_clean_isf(
    folderpath: str, *, time_span: TimeSpan, delta_k: float
) -> None:
    """Run a langevin ensemble and save the resulting clean isf to file."""
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
    )

    result = solve_many_overdamped(
        system,
        time_span,
        initial_conditions=(np.full((500, 1), 0.0), np.full((500, 1), 0.0)),
    )

    isf = get_pairwise_isf(result.x_points, (delta_k,))
    avg_isf = np.mean(isf, axis=0)

    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"
    with Path(clean_isf_path).open("wb") as file:
        real_isf = get_measured_data(avg_isf, "real")
        isf_record = {"time": result.times, "isf": real_isf}
        pickle.dump(isf_record, file)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line,) = ax.plot(result.times, real_isf)
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
    traj_filepath: str, n_traj: int, *, time_span: TimeSpan
) -> None:
    """Run the langevin simulations and save to file."""
    keys = jax.random.split(jax.random.key(100), n_traj)

    # Transfer batched array directly from GPU to CPU as contiguous blocks
    batched_trajectories = jax.tree.map(
        jnp.asarray, run_langevin_trajectories(time_span=time_span, keys=keys)
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

    print("Training model")
    trained_model = train_model(
        training_isf[None, None, :], time_span=time_span, delta_k=delta_k
    )

    print("\nModel trained! Getting model's isf")

    # Test model on clean isf
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, training_isf[None, :]
    )

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
    jnp.array([data.get("isf").get("isf") for data in training_isf_data])
    jnp.array([data.get("isf").get("error") for data in training_isf_data])


if __name__ == "__main__":
    path = "./examples/data"
    many_test(path, 10)
