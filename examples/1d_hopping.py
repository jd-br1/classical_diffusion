import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
    plot_x_evolution,
)
from classical_diffusion.hopping import (
    Lattice1D,
    solve_ensemble,
)
from classical_diffusion.hopping._hopping import (
    get_deterministic_probabilities,
    plot_1d_periodic_deterministic_isf,
    plot_deterministic_isf,
    plot_probability_matrix,
)
from classical_diffusion.plot import (
    get_fancy_figure,
)
from classical_diffusion.simulation import TimeSpan


def _plot_1d_hopping_isf() -> None:

    results = solve_ensemble(
        system=Lattice1D(lattice_spacing=5, hop_time=15),
        time_span=TimeSpan(t_end=400, n_steps=4000),
        initial_condition=np.full((1, 1), 0.0),
        key=jrandom.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()

    fig, ax, _ = plot_x_evolution(result=results, ax=ax)
    ax.set_xlim(0, results.times[-1])
    fig.savefig("./examples/1d_hopping.D_trajectory.pdf")

    results = solve_ensemble(
        system=Lattice1D(lattice_spacing=5, hop_time=15),
        time_span=TimeSpan(t_end=4000, n_steps=4000),
        initial_condition=np.full((4000, 1), 0.0),
        key=jrandom.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / results.system.lattice_spacing,)
    _, ax, line_0, _ = plot_isf(result=results, delta_k=delta_k, ax=ax)
    line_0.set_label("Hopping simulation")
    ax.set_xlim(0, right=25)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/1d_hopping.E_hopping_isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_hopping.E_hopping_isf.log.pdf")


def _plot_1d_deterministic_isf() -> None:

    lattice_parameter = 5
    system = Lattice1D(lattice_spacing=lattice_parameter, hop_time=15)

    result = get_deterministic_probabilities(
        system,
        (501,),
        TimeSpan(t_end=400, n_steps=4000),
        jnp.array([250]),
    )

    fig, ax = get_fancy_figure()
    _, ax = plot_probability_matrix(result, ax=ax)
    ax.set_title("Deterministic Hopping Probability Matrix")
    fig.savefig("./examples/1d_hopping.A_deterministic_probability_matrix.pdf")

    fig, ax = get_fancy_figure()
    delta_k = 0.5 * 2 * np.pi / system.lattice_spacing
    _, ax, line_0 = plot_deterministic_isf(system, result, delta_k, ax=ax)

    line_0.set_label("Deterministic Hopping")
    ax.set_xlim(0, right=50)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/1d_hopping.B_deterministic_isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_hopping.B_deterministic_isf.log.pdf")

    fig, ax = get_fancy_figure()
    _, ax, line_0 = plot_1d_periodic_deterministic_isf(
        system, TimeSpan(t_end=400, n_steps=4000), delta_k, ax=ax
    )

    line_0.set_label("1D Periodic Deterministic Hopping")
    ax.set_xlim(0, right=50)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/1d_hopping.C_1d_periodic_deterministic_isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_hopping.C_1d_periodic_deterministic_isf.log.pdf")


if __name__ == "__main__":
    _plot_1d_deterministic_isf()
    _plot_1d_hopping_isf()
