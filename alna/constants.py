"""
Numerical constants.
"""

from pathlib import Path

from base_models import DATA_PATH, SOLID_EARTH_MODEL_PROFILES
from numpy import abs as numpy_abs
from numpy import arange, array, asarray, concatenate, log, ndarray, pi
from numpy import sum as numpy_sum
from numpy import zeros_like
from scipy.special import zeta
from sympy import Expr, symbols

N_LERCH_SERIES: int = 20
N_JONQUIERE_EXPANSION: int = 8
SMALL_Z_THRESHOLD: float = 0.7
LARGE_Z_THRESHOLD: float = 1.3

### Solid Earth model descriptions.
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH = Path("../alna").joinpath(
    "solid_earth_model_profile_descriptions"
)
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH: dict[str, Path] = {
    model_part: SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH.joinpath(model_part)
    for model_part in SOLID_EARTH_MODEL_PROFILES
}

## Solid Earth numerical models.
SOLID_EARTH_NUMERICAL_MODELS_PATH = DATA_PATH.joinpath("solid_earth_numerical_models")


# Universal Gravitationnal constant (m^3.kg^-1.s^-2).
G = 6.67430e-11

# s.y^-1
SECONDS_PER_YEAR = 365.25 * 86400
COMPLEX_PARTS = ["real", "imag"]


# For integration.
INITIAL_Y_I = (
    array(
        object=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=complex,
    ),
    array(
        object=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dtype=complex,
    ),
    array(
        object=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=complex,
    ),
)
Y_I_STATE_VECTOR_LINE = list(symbols(r"y_1 y_2 y_3 y_4 y_5 y_6"))
Y_I_STATE_FOR_SURFACE: list[list[Expr]] = [
    list(
        symbols(
            r" ".join(
                ("y_" + str(i_line + 1) + "_" + str(i_component + 1) for i_component in range(6))
            )
        )
    )
    for i_line in range(3)
]


def lerch_series(z: complex, s: int, a: float, n: int = N_LERCH_SERIES):
    """
    Computes the Lerch transcendant as a series. Converges for |z| << 1.
    """

    z = asarray(z, dtype=complex)
    k = arange(n)

    return numpy_sum(z[..., None] ** k / (k + a) ** s, axis=-1)


def lerch_jonquiere(
    z: complex,
    s: int,
    a: float,
    k: int = N_JONQUIERE_EXPANSION,
):
    """
    Computes the Lerch transcendant in the asymptotic region |z| ~= 1 of the complex plane for s in
    {1, 2}.
    """

    z = asarray(z, dtype=complex)
    logz = log(z)
    k_tab = arange(k)
    fact = (k_tab > 0).cumprod()  # Factorial via cumulative product.
    fact[0] = 1

    return (-1) ** (s - 1) * sum(zeta(x=s - k_tab, q=a) * logz[..., None] ** k / fact, axis=-1)


def lerch_phi(z: complex, s: int, a: float):
    """
    Lerch Phi trasnscendent function, vectorized and taking advantage of s being an integer for the
    slow convergence area of the complex plane.
    """

    z = asarray(z, dtype=complex)
    r = numpy_abs(z)
    out = zeros_like(z, dtype=complex)
    small: ndarray = r < SMALL_Z_THRESHOLD
    large: ndarray = r > LARGE_Z_THRESHOLD
    mid: ndarray = ~(small | large)

    if small.any():

        out[small] = lerch_series(z=z[small], s=s, a=a)

    if large.any():

        zl = z[large]  # Identity holds assuming no branch cut.
        out[large] = -(1 / zl) * lerch_series(z=1 / zl, s=s, a=a)

    if mid.any():

        out[mid] = lerch_jonquiere(z=z[mid], s=s, a=a)

    return out


def lerch_phi_numpy(z: complex, s: int, a: float):
    """
    Fast numerical equivalences for Lerch transcendant function in special cases s = {0, 1, 2}.
    """

    if s == 0:

        return 1.0 / (1.0 - z)

    if s in [1, 2]:

        return lerch_phi(z=z, s=s, a=a)

    raise NotImplementedError("Only s = {0, 1, 2} supported")


SYMPY_COMPILATION_MODULES_TRANSIENT_FRIENDLY = [
    {"lerchphi": lerch_phi_numpy},
    "numpy",
]

# Other low level parameters.
LAYER_DECIMALS = 5
LAYER_NAMES_SEPARATOR = "__"
SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR = 5 * "_"
SOLID_EARTH_NUMERICAL_MODEL_NAME_FROM_INVERTIBLE_PARAMETERS_SEPARATOR = 2 * "-"
LAYERS_SEPARATOR = "___"
VALUES_SEPARATOR = "__"
UNUSED_MODEL_PART_DEFAULT_NAME = "unused"


def renard_number_system(n_max: int):
    """
    Produces the list [1, 2, 5, 10, 20, ... until n_max.
    """

    result = []
    base = 1

    while base <= n_max:

        for factor in (1, 2, 5):

            value = base * factor

            if value > n_max:

                return result

            result += [value]

        base *= 10

    return result


def generate_degree_tab(n_max: int = 100, n_start_steps: int = 20) -> list[int]:
    """
    Generates the list of degrees for which to compute Love numbers. Applies a Renard number system
    progression.
    """

    initial_list = renard_number_system(n_max=n_max)
    index_start_steps = [
        index for index, value in enumerate(initial_list) if value <= n_start_steps
    ][-1]
    degree_steps = initial_list[:-index_start_steps]
    degree_thresholds = [1] + initial_list[index_start_steps:]

    return concatenate(
        [
            arange(degree_thresholds[i], degree_thresholds[i + 1], degree_step, dtype=int)
            for i, degree_step in enumerate(degree_steps)
        ],
    ).tolist()


def compute_omega_tab(period_tab: ndarray) -> ndarray:
    """
    Pulsation (rad.s^-1) from period (yr).
    """

    return 2 * pi / (SECONDS_PER_YEAR * period_tab)
