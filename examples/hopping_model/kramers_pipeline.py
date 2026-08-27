import os

import jax.numpy as jnp
import optax
from scipy.constants import Boltzmann

from classical_diffusion.hopping import (
    CanonicalLattice,
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
from classical_diffusion.simulation import TimeSpan

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import equinox as eqx
import jax

from classical_diffusion.langevin import (
    CanonicalSystem,  # ruff: ignore[typing-only-first-party-import]
)

jax.config.update("jax_enable_x64", False)


CHECK_EVERY = 2
EARLY_STOP = 0.1  # Improvement to loss over CHECK_EVERY epochs deemed small enough to have reached training plateau
NUM_EPOCHS = 100
BATCH_SIZE = 2  # Number of isfs to test on at a time (does this need to be limited?)
LOSS_BATCH_SIZE = 2  # Number of isfs to calculate loss of at a time


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


@jax.checkpoint
@jax.jit
def get_langevin_isf_directly(
    params: KramersParameters, time_span: TimeSpan, delta_k: float
) -> jnp.ndarray:
    """Get the isf from langevin simulation directly."""
    system = KramersSystem1D(params=params).as_canonical()
    langevin_keys = jax.random.split(jax.random.key(32), 50)
    _, positions = run_langevin_trajectories(
        system=system, time_span=time_span, keys=langevin_keys
    )
    isf = get_pairwise_isf(jnp.array(positions), jnp.asarray((delta_k,)))
    avg_isf = jnp.mean(isf, axis=0)
    return get_measured_data(avg_isf, "real")[0]


@jax.jit
def loss_fn(
    model: eqx.Module,
    *,
    time_span: TimeSpan,
    delta_k: float,
    test_params: jnp.ndarray,
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function\n")

    total_samples = test_params.shape[0]
    num_chunks = total_samples // LOSS_BATCH_SIZE
    usable_samples = num_chunks * LOSS_BATCH_SIZE

    # Pass batched isfs through the model to predict hopping rates and isf offsets
    predictions = jax.vmap(model)(test_params[:usable_samples])  # ty: ignore[invalid-argument-type]
    hopping_times = predictions[:, 0]
    offsets = predictions[:, 1]

    test_params_chunks = test_params[:usable_samples].reshape(
        num_chunks, LOSS_BATCH_SIZE, *test_params[1:]
    )
    hop_time_chunks = hopping_times[:usable_samples].reshape(
        num_chunks, LOSS_BATCH_SIZE
    )

    # Get Langevin isfs
    def _langevin_batch(params_chunk: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(get_langevin_isf_directly, (0, None, None))(
            params_chunk, time_span, delta_k
        )

    # Get hopping isfs
    def _hopping_model_batch(hop_time_chunk: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(get_deterministic_isf_directly, (0, None, None))(
            hop_time_chunk, time_span, delta_k
        )

    # Sequential execution over micro-batches via lax.map
    langevin_isfs = jax.lax.map(_langevin_batch, test_params_chunks)
    hopping_isfs = jax.lax.map(_hopping_model_batch, hop_time_chunks)
    langevin_isfs.reshape(usable_samples, -1)
    hopping_isfs = hopping_isfs.reshape(usable_samples, -1)
    corrected_isfs = offsets[:usable_samples, None] * hopping_isfs

    errors = jnp.sum(
        (corrected_isfs - langevin_isfs) ** 2,
        axis=-1,
    )

    # Return the average error
    return jnp.mean(errors)


# Training functions


def train_model(
    training_data: jnp.ndarray,
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

    training_params = training_data

    # Define number batches per epoch
    num_params = len(training_params)
    num_batches = num_params // BATCH_SIZE
    usable_len = num_batches * BATCH_SIZE

    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def batch_step(
        carry: tuple[eqx.Module, optax.OptState], batch: jnp.ndarray
    ) -> tuple[tuple[eqx.Module, optax.OptState], jnp.ndarray]:
        model, opt_state = carry
        batch_params = batch

        batch_loss, gradients = loss_and_grad_fn(
            model,
            time_span=time_span,
            delta_k=delta_k,
            test_params=batch_params,
        )

        updates, opt_state = optimizer.update(gradients, opt_state)

        model = eqx.apply_updates(model, updates)

        return (model, opt_state), batch_loss

    @eqx.filter_jit
    def run_epoch(
        model: eqx.Module, opt_state: optax.OptState, params: jnp.ndarray
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        """Run one epoch of ML algorithm."""
        batched_params = params[:usable_len].reshape(
            num_batches, BATCH_SIZE, *params.shape[1:]
        )

        (model, opt_state), batch_losses = jax.lax.scan(
            batch_step, (model, opt_state), (batched_params)
        )

        return model, opt_state, jnp.sum(batch_losses)

    # Control loop outside jit
    losses = []
    for epoch in range(NUM_EPOCHS):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_params)

        shuffled_params = training_params[permutation]

        model, optimizer_state, epoch_loss = run_epoch(
            model, optimizer_state, shuffled_params
        )
        losses.append(float(epoch_loss))

        if (epoch + 1) % CHECK_EVERY == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss:.5f}")
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


@jax.jit
def _inits_constant() -> tuple[
    jnp.ndarray,
    jnp.ndarray,
]:
    const_initial_cond = jnp.full((1, 1), 0.0)
    return (const_initial_cond, const_initial_cond)


@eqx.filter_jit
def run_langevin_trajectories(
    *,
    system: CanonicalSystem,
    time_span: TimeSpan,
    keys: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run langevin trajectories."""

    def body(time_span: TimeSpan, _key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:

        initial_conditions = _inits_constant(1)

        times, positions, _ = solve_many_overdamped_jax(
            system,
            time_span,
            initial_conditions,
            _key=_key,
        )

        return times, positions

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


def derive_rate_theory() -> None:
    """Train a model on a range of omega_well values and correlate ."""
    print("\n\nRunning kamers rate theory test\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    delta_k = 0.5

    # Generate parameter arrays - for now just vary barrier energy
    def new_params(_key: jax.Array):

        rand = jax.random.uniform(_key, shape=(), minval=2.0, maxval=5.0)
        return KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=rand.astype(jnp.float32),  # ty: ignore[invalid-argument-type]
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )

    print("\nGenerating parameters")
    num_params = 20
    keys = jax.random.split(jax.random.key(77), num_params)
    parameter_array = jax.vmap(new_params, (0))(keys)

    print(parameter_array)
    model = train_model(
        training_data=parameter_array, time_span=time_span, delta_k=delta_k
    )

    print("\nModel trained :(")
    test_barrier_energies = jnp.linspace(2.0, 5.0, 6)

    hopping_times = []
    for test_barrier_energy in test_barrier_energies:
        params = KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=test_barrier_energy,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
        prediction = model(jnp.asarray(params))
        hopping_times.append(prediction[0])

    print(test_barrier_energies)
    print(hopping_times)


if __name__ == "__main__":
    derive_rate_theory()
