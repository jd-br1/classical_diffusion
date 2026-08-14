import jax
import jax.numpy as jnp
import numpy as np

from classical_diffusion.hopping import (
    DeterministicSolverResult,
    Lattice,
    Lattice1D,
    get_ensemble_probabilities,
    plot_deterministic_isf,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import timed


@jax.jit
def _get_jensen_probabilities_jit(
    initial_p: jnp.ndarray,
    final_time: jax.Array,
    hop_sites: jnp.ndarray,
    hop_rates: jnp.ndarray,
) -> jnp.ndarray:
    """Use deterministic formula to return the ISF, efficiently."""
    total_outgoing_rates = jnp.sum(hop_rates, axis=-1)

    # Determine uniform rate gamma (must be >= max outgoing rate)
    gamma = jnp.max(total_outgoing_rates) * 1.05

    # Carry: (current term's probability contribution, accumulated terms' probability contribution, possion weighting)
    init_carry = (initial_p, jnp.zeros(initial_p.shape), -gamma * final_time)

    # Iteration step n
    def iter_step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jax.Array], n: jax.Array
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jax.Array], None]:
        nth_term_probabilities, cumulative_probabilities, log_poisson_weight = carry

        # Accumulate weighted probability
        poisson_weight = jnp.exp(log_poisson_weight)
        cumulative_probabilities += poisson_weight * nth_term_probabilities

        dp_dt = (
            jnp.sum(hop_rates * nth_term_probabilities[hop_sites], axis=-1)
            - nth_term_probabilities * total_outgoing_rates
        )
        n_plus_1th_term_probabilities = nth_term_probabilities + (1.0 / gamma) * dp_dt

        next_log_poisson_weight = (
            log_poisson_weight + jnp.log(gamma * final_time) - jnp.log(n + 1.0)
        )

        return (
            n_plus_1th_term_probabilities,
            cumulative_probabilities,
            next_log_poisson_weight,
        ), None

    ns = jnp.arange(30, dtype=jnp.float32)
    (_, total_probabilities, _), _ = jax.lax.scan(iter_step, init_carry, ns)

    return total_probabilities


@timed
def get_jensen_probabilities[L: Lattice](
    system: L,
    lattice_sizes: tuple[int, ...],
    time_span: TimeSpan,
    initial_position: int,
) -> DeterministicSolverResult:
    """Use the Jensen Solver to find the ensemble probabilities at all times."""
    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps)

    initial_p = jnp.full(np.prod(lattice_sizes), 0.0, dtype=jnp.float32)
    initial_p = initial_p.at[initial_position].set(1)

    hop_sites, hop_rates = system.get_rates(np.arange(np.prod(lattice_sizes)))

    # Calculate a sensible bound - can't do this in jit function, so print to check it's about right
    # Need to find a way to allocate the correct amount of memory for this calculated bound in jax
    total_outgoing_rates = jnp.sum(hop_rates, axis=-1)
    gamma = jnp.max(total_outgoing_rates) * 1.05
    lam = gamma * times[-1]
    print(lam + 2 * jnp.sqrt(lam) + 10)

    sol = _get_jensen_probabilities_jit(initial_p, times[-1], hop_sites, hop_rates)

    return DeterministicSolverResult(
        system=system, times=np.array(times), probabilities=np.array(sol)
    )


def _deterministic_solvers_benchmark() -> None:

    lattice = Lattice1D(lattice_spacing=5, hop_time=15)
    max_lattice_size = 10000
    single_time_step = TimeSpan(t_end=100, n_steps=50)
    initial_pos = int(0.5 * max_lattice_size)

    deterministic_results = get_ensemble_probabilities(
        lattice, (max_lattice_size + 1,), single_time_step, initial_pos
    )

    jensen_results = get_jensen_probabilities(
        lattice, (max_lattice_size + 1,), single_time_step, initial_pos
    )

    print(deterministic_results.probabilities[-1][initial_pos - 5 : initial_pos + 5])
    print(jensen_results.probabilities[initial_pos - 5 : initial_pos + 5])

    fig, ax = get_fancy_figure()
    delta_k = 0.5 * 2 * np.pi / lattice.lattice_spacing
    _, ax, line_0 = plot_deterministic_isf(
        lattice, deterministic_results, delta_k, ax=ax
    )

    line_0.set_label("Deterministic Hopping")
    ax.set_xlim(0, right=25)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/test.isf.deterministic.pdf")


if __name__ == "__main__":
    _deterministic_solvers_benchmark()
