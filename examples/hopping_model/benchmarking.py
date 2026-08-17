import numpy as np

from classical_diffusion.hopping import (
    Lattice1D,
    deterministic,
    get_ensemble_probabilities,
    jensen,
    plot_deterministic_isf,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _deterministic_solvers_benchmark() -> None:

    lattice = Lattice1D(lattice_spacing=5, hop_time=15)
    times = TimeSpan(t_end=100, n_steps=1000)

    print("\n1000 lattice points")
    get_ensemble_probabilities(lattice, (1001,), times, 500, deterministic)
    print("\n10000 lattice points")
    get_ensemble_probabilities(lattice, (10001,), times, 5000, deterministic)
    print("\n100000 lattice points")
    deterministic_results = get_ensemble_probabilities(
        lattice,
        (100001,),
        times,
        50000,
        deterministic,
    )

    print("\n\nJensen")
    print("\n1000 lattice points")
    get_ensemble_probabilities(lattice, (1001,), times, 500, jensen)
    print("\n10000 steps")
    get_ensemble_probabilities(lattice, (10001,), times, 5000, jensen)
    print("\n100000 steps")
    jensen_results = get_ensemble_probabilities(
        lattice, (100001,), times, 50000, jensen
    )

    print(deterministic_results.probabilities[-1][50000 - 5 : 50000 + 5])
    print(jensen_results.probabilities[-1][50000 - 5 : 50000 + 5])

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
