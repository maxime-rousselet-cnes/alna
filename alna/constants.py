"""
Numerical constants.
"""

from pathlib import Path

from base_models import DATA_PATH, SOLID_EARTH_MODEL_PROFILES, TEST_PATH
from numpy import arange, array, concatenate, exp, ndarray, pi
from numpy.polynomial.laguerre import laggauss
from sympy import Expr, symbols

N_GAUSS_LAGUERRE = 64
N_LERCH_SERIES = 50
T_GAUSS_LAGUERRE, W_GAUSS_LAGUERRE = laggauss(deg=N_GAUSS_LAGUERRE)

### Solid Earth model descriptions.
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH = Path("../alna").joinpath(
    "solid_earth_model_profile_descriptions"
)
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH: dict[str, Path] = {
    model_part: SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH.joinpath(model_part)
    for model_part in SOLID_EARTH_MODEL_PROFILES
}

TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH = TEST_PATH.joinpath("solid_earth_numerical_models")
TEST_ELASTIC_INTEGRATION_PATH = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath(
    "elastic_integration_test"
)

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
    Direct series for |z| < 1.
    """

    zn = 1.0 + 0j
    out = 0.0 + 0j

    for k in range(n):

        out += zn / ((k + a) ** s)
        zn *= z

    return out + zn / (1 - z) / ((n + a) ** s)


def lerch_integral(z: complex, s: int, a: float, t=T_GAUSS_LAGUERRE, w=W_GAUSS_LAGUERRE):
    """
    Gauss-Laguerre integral form for general z.
    """

    denom = 1.0 - z * exp(-t)

    if s == 1:

        f = exp(-(a - 1) * t) / denom

    elif s == 2:

        f = t * exp(-(a - 1) * t) / denom

    else:

        raise ValueError

    return sum(w * f)


def lerch(z: complex, s: int, a: float, n: int = N_LERCH_SERIES):
    """
    Lerch transcendent for s in {0, 1, 2}.
    """

    if s == 0:

        return 1.0 / (1.0 - z)

    if abs(z) < 0.8:

        return lerch_series(z=z, s=s, a=a, n=n)

    return lerch_integral(z=z, s=s, a=a)


SYMPY_COMPILATION_MODULES_TRANSIENT_FRIENDLY = [
    {"lerchphi": lerch},
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
