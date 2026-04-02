"""
Formulas involving rheological constants.
"""

from numpy import array, ndarray, pi
from sympy import (
    DotProduct,
    Expr,
    I,
    Matrix,
    MutableDenseMatrix,
    Piecewise,
    exp_polar,
    integrate,
    log,
)
from sympy import pi as sympy_pi
from sympy import symbols
from sympy.core.numbers import Infinity, One, Zero

from .parameters import DEFAULT_COMPONENT_PARAMETERS, ComponentParameters


def mu_0_computing(rho_0: Expr, v_s: Expr) -> Expr:
    """
    Computes real elastic modulus mu given density rho_0 and S wave speed v_s.
    """

    return rho_0 * v_s**2


def lambda_0_computing(rho_0: Expr, v_p: Expr, mu_0: Expr) -> Expr:
    """
    Computes real elastic modulus lambda given density rho_0, P wave speed v_p and real elastic
    modulus mu.
    """

    return rho_0 * v_p**2 - 2 * mu_0


def g_0_computing(x: Expr, rho_0: Expr, g_0_inf: Expr = Zero(), x_inf: float = 0.0) -> Expr:
    """
    Integrates the internal mass GM to get gravitational acceleration g.
    Assumes the nondimensionalization results in pi * G = 1.
    """

    upper_integral: Expr = 4 * integrate(rho_0 * x**2, (x, x_inf, x))
    lower_integral: Expr = x**2 * g_0_inf

    return (upper_integral + lower_integral.xreplace(rule={x: x_inf})) / x**2


def mu_k_computing(mu_k1: Expr, c: Expr, mu_0: Expr) -> Expr:
    """
    Computes Kelvin's equivalent elastic modulus given the parameters mu_{k1}, c, and real elastic
    modulus mu_0.
    """

    return mu_k1 + c * mu_0


def characteristic_pulsation_computing(mu: Expr, eta: Expr) -> Expr:
    """
    Computes characteristic frequency value given the real elastic modulus mu and viscosity eta.
    """

    return mu / eta


def m_prime_computing(maxwell_characteristic_pulsation: Expr, omega: Expr) -> Expr:
    """
    Computes m' transfert function value given the Maxwell's characteristic frequency and pulsation
    value omega * j.
    """

    return maxwell_characteristic_pulsation / (maxwell_characteristic_pulsation + omega * I)


def b_computing(
    maxwell_characteristic_pulsation: Expr,
    kelvin_characteristic_pulsation: Expr,
    burgers_characteristic_pulsation: Expr,
    omega: Expr,
) -> Expr:
    """
    Computes b transfert function value given the Maxwell's, Kelvin's and Burgers characteristic
    frequencies and pulsation value omega.
    """

    return (omega * I * burgers_characteristic_pulsation) / (
        (omega * I + kelvin_characteristic_pulsation)
        * (omega * I + maxwell_characteristic_pulsation)
    )


def unbounded_f_attenuation_computing(
    frequency: Expr, alpha: Expr, frequency_unit: float, omega_m_inf: Expr
) -> Expr:
    """
    Computes the unbounded (short-term approximation) attenuation function f using parameters
    omega_m_inf and alpha. Variables frequency and omega_m_inf should represent unitless
    frequencies.
    """

    omega_0 = 1.0  # (Hz).

    return Piecewise(
        (2.0 / pi * log((frequency * frequency_unit) / omega_0) + 1.0j, frequency >= omega_m_inf),
        (
            2.0
            / pi
            * (
                log(omega_m_inf * frequency_unit)
                + 1 / alpha * (1 - (omega_m_inf / frequency) ** alpha)
            )
            + 1j * (omega_m_inf / frequency) ** alpha,
            True,
        ),
    )


def find_tau_m_sup(
    omega_m_inf: Expr, period_unit: float, alpha: Expr, delta: Expr, q_mu: Expr
) -> Expr:
    """
    Uses asymptotic equation to find tau_m_sup such as it is constrained by the relative amplitude
    of the transient response compared to the elastic response.
    """

    frequency_unit = 1 / period_unit
    tau_0 = 1.0  # (s).

    return ((omega_m_inf * frequency_unit) ** (-alpha) + alpha * delta * q_mu * tau_0**alpha) ** (
        1.0 / alpha
    ) / period_unit


def rewrite_alpha_integral(expression: Expr) -> Expr:
    """
    Needed to rewrite the automatically derived expresison from sympy.
    """

    return expression.xreplace(rule={exp_polar(3 * I * sympy_pi / 2): -I}).simplify()


def bounded_f_attenuation_computing(
    omega: Expr,
    alpha: Expr,
    omega_m_inf: Expr,
    tau_m_sup: Expr,
    period_unit: float,
) -> Expr:
    """
    Computes the bounded (full-domain valid) attenuation function f. The variable tau_m_sup should
    represent a unitless time, omega_m_inf a unitless frequency and omega a unitless pulsation.
    """

    tau_0 = 1.0  # (s).
    frequency_unit = 1 / period_unit
    tau, frequency_unit_symbol = symbols(r"\tau f_{unit}")
    tau_m_inf = 1 / omega_m_inf

    f = (
        Zero()
        if alpha == 1 or tau_m_inf == tau_m_sup
        else -rewrite_alpha_integral(  # TODO: + or -? Relaunch with +.
            expression=integrate(
                (tau / tau_0) ** alpha / (1 + I * omega * frequency_unit_symbol * tau) / tau,
                (tau, tau_m_inf * period_unit, tau_m_sup * period_unit),
            ),
        ).xreplace(rule={frequency_unit_symbol: frequency_unit})
    )

    return f


def delta_mu_computing(mu_0: Expr, q_mu: Expr, f_attenuation: Expr) -> Expr:
    """
    Computes the complex frequency dependent anelastic shear modulus deviation from elasticity.
    """

    return (mu_0 / q_mu) * f_attenuation


def mu_computing(mu_0: Expr, q_mu: Expr, f_attenuation: Expr, m_prime: Expr, b: Expr) -> Expr:
    """
    Computes the complex analog mu value from available profiles.
    """

    delta_mu = delta_mu_computing(
        mu_0=mu_0,
        q_mu=q_mu,
        f_attenuation=f_attenuation,
    )

    return (mu_0 + delta_mu) * (1 - m_prime) / (1 + b)


def attenuation_function_computing(
    expressions: dict[str, Expr],
    units: dict[str, float],
    bounded_attenuation_functions: bool = True,
) -> Expr:
    """
    Attenuation function formula, whether it is bounded or not.
    """

    return (
        unbounded_f_attenuation_computing(
            frequency=expressions[r"\omega"] / (2 * pi),
            alpha=expressions[r"\alpha"],
            frequency_unit=units[r"f"],
            omega_m_inf=expressions[r"\omega_{m-inf}"],
        )
        if not bounded_attenuation_functions
        else bounded_f_attenuation_computing(
            omega=expressions[r"\omega"],
            alpha=expressions[r"\alpha"],
            omega_m_inf=expressions[r"\omega_{m-inf}"],
            tau_m_sup=find_tau_m_sup(
                omega_m_inf=expressions[r"\omega_{m-inf}"],
                period_unit=units[r"T"],
                alpha=expressions[r"\alpha"],
                delta=expressions[r"\Delta"],
                q_mu=expressions[r"q_\mu"],
            ),
            period_unit=units[r"T"],
        )
    )


def create_rheological_expressions(
    expressions: dict[str, Expr],
    units: dict[str, float],
    component_parameters: ComponentParameters = DEFAULT_COMPONENT_PARAMETERS,
) -> dict[str, Expr]:
    """
    Creates all the needed rheological expressions from their polynomials.
    """

    expressions[r"\mu_0"] = mu_0_computing(rho_0=expressions[r"\rho_0"], v_s=expressions[r"v_s"])
    expressions[r"\lambda_0"] = lambda_0_computing(
        rho_0=expressions[r"\rho_0"], v_p=expressions[r"v_p"], mu_0=expressions[r"\mu_0"]
    )

    if not component_parameters.viscous_component:

        m_prime = Zero()
        b = Zero()

    else:

        maxwell_characteristic_pulsation = characteristic_pulsation_computing(
            mu=expressions[r"\mu_0"], eta=expressions[r"\eta_m"]
        )
        m_prime = m_prime_computing(
            maxwell_characteristic_pulsation=maxwell_characteristic_pulsation,
            omega=expressions[r"\omega"],
        )
        b = (
            Zero()
            if r"\eta_k" not in expressions
            else b_computing(
                maxwell_characteristic_pulsation=maxwell_characteristic_pulsation,
                kelvin_characteristic_pulsation=characteristic_pulsation_computing(
                    mu=mu_k_computing(
                        mu_k1=expressions[r"\mu_{k1}"],
                        c=expressions[r"c"],
                        mu_0=expressions[r"\mu_0"],
                    ),
                    eta=expressions[r"\eta_k"],
                ),
                burgers_characteristic_pulsation=characteristic_pulsation_computing(
                    mu=expressions[r"\mu_0"], eta=expressions[r"\eta_k"]
                ),
                omega=expressions[r"\omega"],
            )
        )

    if not component_parameters.transient_component:

        expressions[r"q_\mu"] = Infinity()
        expressions[r"f_{attenuation}"] = Zero()

    else:

        expressions[r"f_{attenuation}"] = attenuation_function_computing(
            expressions=expressions,
            units=units,
            bounded_attenuation_functions=component_parameters.bounded_attenuation_functions,
        )

    expressions[r"\mu_{complex}"] = mu_computing(
        mu_0=expressions[r"\mu_0"],
        q_mu=expressions[r"q_\mu"],
        f_attenuation=expressions[r"f_{attenuation}"],
        m_prime=m_prime,
        b=b,
    )
    # Hypothesis that compressibility is reduced to its elastic component.
    expressions[r"\lambda_{complex}"] = expressions[r"\lambda_0"] - 2.0 / 3.0 * (
        expressions[r"\mu_{complex}"] - expressions[r"\mu_0"]
    )

    return expressions


def fluid_system_matrix(expressions: dict[str, Expr]) -> MutableDenseMatrix:
    """
    Defines the fluid integration system matrix.
    """

    # Smylie (2013) Eq.9.42 & 9.43.
    c_1_1 = 4.0 * expressions[r"\rho_0"] / expressions[r"g_0"]

    return MutableDenseMatrix(
        [
            [c_1_1, One()],
            [
                (expressions[r"n"] * (expressions[r"n"] + 1.0) / expressions[r"x"] ** 2)
                - 16.0 * expressions[r"\rho_0"] / (expressions[r"g_0"] * expressions[r"x"]),
                (-2.0 / expressions[r"x"]) - c_1_1,
            ],
        ]
    )


def solid_system_matrix(
    expressions: dict[str, Expr], components: ComponentParameters = DEFAULT_COMPONENT_PARAMETERS
) -> MutableDenseMatrix:
    """
    Defines the solid integration system matrix.
    """

    # Intermediate variables for lisibility.
    dyn_term = (
        -expressions[r"\rho_0"] * expressions[r"\omega"] ** 2.0
        if components.viscous_component or components.transient_component
        else Zero()
    )
    n_1 = expressions[r"n"] * (expressions[r"n"] + 1.0)
    b = 1.0 / (expressions[r"\lambda_{complex}"] + 2.0 * expressions[r"\mu_{complex}"])
    c = (
        2.0
        * expressions[r"\mu_{complex}"]
        * (3.0 * expressions[r"\lambda_{complex}"] + 2.0 * expressions[r"\mu_{complex}"])
        * b
    )

    return MutableDenseMatrix(
        [
            [
                -2.0 * expressions[r"\lambda_{complex}"] * b / expressions[r"x"],
                b,
                n_1 * expressions[r"\lambda_{complex}"] * b / expressions[r"x"],
                Zero(),
                Zero(),
                Zero(),
            ],
            [
                (-4.0 * expressions[r"g_0"] * expressions[r"\rho_0"] / expressions[r"x"])
                + (2.0 * c / expressions[r"x"] ** 2)
                + dyn_term,
                -4.0 * expressions[r"\mu_{complex}"] * b / expressions[r"x"],
                n_1
                * (
                    expressions[r"\rho_0"] * expressions[r"g_0"] / expressions[r"x"]
                    - c / (expressions[r"x"] ** 2)
                ),
                n_1 / expressions[r"x"],
                Zero(),
                -expressions[r"\rho_0"],
            ],
            [
                -1.0 / expressions[r"x"],
                Zero(),
                1.0 / expressions[r"x"],
                1.0 / expressions[r"\mu_{complex}"],
                Zero(),
                Zero(),
            ],
            [
                expressions[r"\rho_0"] * expressions[r"g_0"] / expressions[r"x"]
                - c / expressions[r"x"] ** 2,
                -expressions[r"\lambda_{complex}"] * b / expressions[r"x"],
                (
                    4.0
                    * n_1
                    * expressions[r"\mu_{complex}"]
                    * (expressions[r"\lambda_{complex}"] + expressions[r"\mu_{complex}"])
                    * b
                    - 2.0 * expressions[r"\mu_{complex}"]
                )
                / expressions[r"x"] ** 2
                + dyn_term,
                -3.0 / expressions[r"x"],
                -expressions[r"\rho_0"] / expressions[r"x"],
                Zero(),
            ],
            [4.0 * expressions[r"\rho_0"], Zero(), Zero(), Zero(), Zero(), One()],
            [
                Zero(),
                Zero(),
                -4.0 * expressions[r"\rho_0"] * n_1 / expressions[r"x"],
                Zero(),
                n_1 / expressions[r"x"] ** 2,
                -2.0 / expressions[r"x"],
            ],
        ]
    )


def solid_to_fluid(
    y_1: ndarray, y_2: ndarray, y_3: ndarray, rho_0_fluid_inf: float, g_0_fluid_inf: float
) -> ndarray:
    """
    Converts the y_i system solution at a fluid/solid interface.
    To call for the first fluid layer.
    """

    k_1_3 = y_1[3] / y_3[3]
    k_2_3 = y_2[3] / y_3[3]
    k_numerator = (
        g_0_fluid_inf * (y_1[0] + y_3[0] * k_1_3)
        - (y_1[4] + y_3[4] * k_1_3)
        + (1.0 / rho_0_fluid_inf) * (y_1[1] + y_3[1] * k_1_3)
    )
    k_denominator = (
        g_0_fluid_inf * (y_2[0] + y_3[0] * k_2_3)
        - (y_2[4] + y_3[4] * k_2_3)
        + (1.0 / rho_0_fluid_inf) * (y_2[1] + y_3[1] * k_2_3)
    )
    k_k = k_numerator / k_denominator
    sol_2 = y_1[1] + k_k * y_2[1] + (k_1_3 + k_k * k_2_3) * y_3[1]
    sol_5 = y_1[4] + k_k * y_2[4] + (k_1_3 + k_k * k_2_3) * y_3[4]
    sol_6 = y_1[5] + k_k * y_2[5] + (k_1_3 + k_k * k_2_3) * y_3[5]

    return array([sol_5, sol_6 + (4.0 / g_0_fluid_inf) * sol_2], dtype=complex)


def fluid_to_solid(
    yf_1: ndarray, rho_0_fluid_sup: float, g_0_fluid_sup: float
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Converts the y_i system solution at a fluid/solid interface.
    """

    return (
        array(
            [1.0, g_0_fluid_sup * rho_0_fluid_sup, 0.0, 0.0, 0.0, -4.0 * rho_0_fluid_sup],
            dtype=complex,
        ),
        array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=complex),
        array([yf_1[0] / g_0_fluid_sup, 0.0, 0.0, 0.0, yf_1[0], yf_1[1]], dtype=complex),
    )


def surface_solution(
    n: int,
    y_1_s: list[Expr],
    y_2_s: list[Expr],
    y_3_s: list[Expr],
    g_0_surface: Expr,
) -> MutableDenseMatrix:
    """
    Returns load Love numbers from the y_i system solution at Earth surface.
    To call for the very last layer.
    """

    surface_factor_1 = (2.0 * n + 1.0) * g_0_surface
    surface_factor_2 = surface_factor_1 * g_0_surface / 4.0

    # Forms the outer surface vectors. See Okubo & Saito (1983), Saito (1978).
    d_marix_load = MutableDenseMatrix([[-surface_factor_2, Zero(), surface_factor_1]])
    d_matrix_shear = MutableDenseMatrix([[Zero(), surface_factor_2 / (n * (n + 1)), Zero()]])
    d_matrix_potential = MutableDenseMatrix([[Zero(), Zero(), surface_factor_1]])

    # Forms the G matrix from integrated solutions.
    g_matrix_inv = MutableDenseMatrix(
        [
            [y_1_s[1], y_2_s[1], y_3_s[1]],
            [y_1_s[3], y_2_s[3], y_3_s[3]],
            [
                (y_1_s[5] + (n + 1.0) * y_1_s[4]),
                (y_2_s[5] + (n + 1.0) * y_2_s[4]),
                (y_3_s[5] + (n + 1.0) * y_3_s[4]),
            ],
        ]
    ).inv()

    # Love numbers.
    love = []

    # Iterates on boundary conditions.
    for d_matrix in [d_marix_load] + ([] if n == 1 else [d_matrix_shear, d_matrix_potential]):

        # Solves the system.
        m_vector = Matrix(g_matrix_inv @ d_matrix.T)

        # Computes solutions.
        love += [
            DotProduct(MutableDenseMatrix([y_1_s[0], y_2_s[0], y_3_s[0]]), m_vector),
            DotProduct(MutableDenseMatrix([y_1_s[2], y_2_s[2], y_3_s[2]]), m_vector),
            DotProduct(MutableDenseMatrix([y_1_s[4], y_2_s[4], y_3_s[4]]), m_vector) / g_0_surface
            - 1.0,  # Because k + 1 is solution.
        ]

    love = MutableDenseMatrix(love)

    # Transforms to the isomorphic frame (Blewitt, 2003) for which the potential field outside
    # the Earth vanishes (e.g. Merriam 1985).
    if n == 1:

        return MutableDenseMatrix(  # Substracts k_load from h_load and l_load.
            [
                [love[0] - love[2], love[1] - love[2], Zero()],
                [Zero(), Zero(), Zero()],
                [Zero(), Zero(), Zero()],
            ]
        )

    love[5] += 1.0  # No shear component on the unperturbed potential.

    return love.reshape(rows=3, cols=3)
