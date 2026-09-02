import jax.numpy as jnp
from scipy.constants import Boltzmann


def get_isf_offset(
    params: jnp.ndarray, delta_k: float, long_time: float
) -> jnp.ndarray:

    omega_well = params[0]
    m = params[3]
    kbt = params[4] * Boltzmann
    gamma = params[5]

    f = jnp.sqrt(omega_well**2 - gamma**2 / 4)

    return jnp.exp(
        -(delta_k**2)
        * (kbt / (m * omega_well**2))
        * (
            1
            - jnp.exp(-gamma * long_time / 2)
            * (jnp.cos(f * long_time) + (gamma / (2 * f)) * jnp.sin(f * long_time))
        )
    )
