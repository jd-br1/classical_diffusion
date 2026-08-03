import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
    plot_x_evolution,
    plot_xy_trajectory,
)
from classical_diffusion.hopping import (
    solve_ensemble,
)
from classical_diffusion.hopping._system import Lattice2D
from classical_diffusion.plot import (
    get_fancy_figure,
)
from classical_diffusion.simulation import TimeSpan


def _plot_2d_hopping_isf() -> None:

    results = solve_ensemble(
        system=Lattice2D(lattice_spacing=5, hop_time=15),
        time_span=TimeSpan(t_end=500, n_steps=1000),
        initial_condition=np.full((1, 2), 0.0),
        key=jrandom.PRNGKey(seed=105),
    )
    fig, ax = get_fancy_figure()

    fig, ax, _ = plot_x_evolution(result=results, ax=ax, idx=0)
    ax.set_xlim(0, results.times[-1])
    fig.savefig("./examples/2d_hopping.A_x_trajectory.pdf")

    fig, ax = get_fancy_figure()

    fig, ax, _ = plot_x_evolution(result=results, ax=ax, idx=1)
    ax.set_xlim(0, results.times[-1])
    fig.savefig("./examples/2d_hopping.B_y_trajectory.pdf")

    fig, ax = get_fancy_figure()

    fig, ax, _ = plot_xy_trajectory(result=results, ax=ax)
    fig.savefig("./examples/2d_hopping.C_xy_trajectory.pdf")

    results = solve_ensemble(
        system=Lattice2D(lattice_spacing=5, hop_time=15),
        time_span=TimeSpan(t_end=500, n_steps=1000),
        initial_condition=np.full((4000, 2), 0.0),
        key=jrandom.PRNGKey(seed=105),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / results.system.lattice_spacing,)
    _, ax, line_0, _ = plot_isf(result=results, delta_k=delta_k, ax=ax)
    line_0.set_label("Hopping simulation")
    ax.set_xlim(0, right=25)
    ax.set_ylim(0, 1)

    fig.savefig("./examples/2d_hopping.D_hopping_isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/2d_hopping.D_hopping_isf.log.pdf")


if __name__ == "__main__":
    _plot_2d_hopping_isf()
