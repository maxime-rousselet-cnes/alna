"""
Numerical constants.
"""

from pathlib import Path

from base_models import DATA_PATH, SOLID_EARTH_MODEL_PROFILES
from numpy import arange, array, asarray, concatenate, ndarray, ones_like, pi, zeros_like
from sympy import Expr, symbols

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


def lerchphi_s1(z: complex, a: float, tol: float = 1e-12, maxiter: int = 200):
    """
    Numerical approximation of the Lerch transcendant function for special case s = 2.
    """

    z = asarray(z, dtype=complex)
    out = zeros_like(z, dtype=complex)
    absz = abs(z)
    mask_small = absz < 0.8

    if any(mask_small):

        zz = z[mask_small]
        zk = ones_like(zz)
        s = zk / a

        for k in range(1, maxiter):

            zk *= zz
            term = zk / (k + a)
            s += term

            if max(abs(term)) < tol:

                break

        out[mask_small] = s

    mask_large = absz > 1.2

    if any(mask_large):

        zz = z[mask_large]
        w = 1.0 / zz
        zk = ones_like(w)
        s = zk / a

        for k in range(1, maxiter):

            zk *= w
            term = zk / (k + a)
            s += term

            if max(abs(term)) < tol:

                break

        out[mask_large] = (zz ** (-a)) * s

    mask_mid = ~(mask_small | mask_large)

    if any(mask_mid):

        zz = z[mask_mid]
        zk = ones_like(zz)
        s = zk / a

        for k in range(1, 60):  # Fixed cutoff for stability.

            zk *= zz
            s += zk / (k + a)

        out[mask_mid] = s

    return out


def lerchphi_s2(z: complex, a: float, tol: float = 1e-12, maxiter: int = 200):
    """
    Numerical approximation of the Lerch transcendant function for special case s = 2.
    """

    z = asarray(z, dtype=complex)
    out = zeros_like(z, dtype=complex)
    absz = abs(z)
    mask_small = absz < 0.8

    if any(mask_small):

        zz = z[mask_small]
        zk = ones_like(zz)
        s = zk / (a**2)

        for k in range(1, maxiter):

            zk *= zz
            term = zk / (k + a) ** 2
            s += term

            if max(abs(term)) < tol:

                break

        out[mask_small] = s

    mask_large = absz > 1.2

    if any(mask_large):

        zz = z[mask_large]
        w = 1.0 / zz
        zk = ones_like(w)
        s = zk / (a**2)

        for k in range(1, maxiter):

            zk *= w
            term = zk / (k + a) ** 2
            s += term

            if max(abs(term)) < tol:

                break

        out[mask_large] = (zz ** (-a)) * s

    mask_mid = ~(mask_small | mask_large)

    if any(mask_mid):

        zz = z[mask_mid]
        zk = ones_like(zz)
        s = zk / (a**2)

        for k in range(1, 80):  # Slightly longer for s = 2.

            zk *= zz
            s += zk / (k + a) ** 2

        out[mask_mid] = s

    return out


def lerchphi_numpy(z, s, a):
    """
    Fast numerical equivalences for Lerch transcendant function in special cases.
    """

    if s == 0:

        return 1.0 / (1.0 - z)

    if s == 1:

        return lerchphi_s1(z, a)

    if s == 2:

        return lerchphi_s2(z, a)

    raise NotImplementedError("Only s = {0, 1, 2} supported")


SYMPY_COMPILATION_MODULES_TRANSIENT_FRIENDLY = [
    {"lerchphi": lerchphi_numpy},
    "numpy",
    {"hyper": lambda *args: (_ for _ in ()).throw(NotImplementedError("Unexpected hyper"))},
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
