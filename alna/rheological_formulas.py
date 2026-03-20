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
    Symbol,
    integrate,
    log,
    symbols,
)
from sympy.core.numbers import One, Zero


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
    Assumes the adimensionalization results in pi * G = 1.
    """

    return 4.0 * integrate(rho_0 * x**2, x) / x**2 + g_0_inf.xreplace({x: x_inf})


def p_0_computing(
    x: Expr, rho_0: Expr, g_0: Expr, p_0_inf: Expr = Zero(), x_inf: float = 0.0
) -> Expr:
    """
    Integrates the static equation to get P(x) = C^te - integral(rho_0 g dx).
    """
    # TODO: verify the surface evaluation is substracted from every layer's p_0 expression.

    return p_0_inf.xreplace({x: x_inf}) - integrate(rho_0 * g_0, x)


def mu_k_computing(mu_k1: Expr, c: Expr, mu_0: Expr) -> Expr:
    """
    Computes Kelvin's equivalent elastic modulus given the parameters mu_k1, c, and real elastic
    modulus mu_0.
    """

    return mu_k1 + c * mu_0


def characteristic_frequency_computing(mu: Expr, eta: Expr) -> Expr:
    """
    Computes characteristic frequency value given the real elastic modulus mu and viscosity eta.
    """

    return mu / eta


def m_prime_computing(maxwell_characteristic_frequency: Expr, omega: Expr) -> Expr:
    """
    Computes m' transfert function value given the Maxwell's characteristic frequency and pulsation
    value omega * j.
    """

    return maxwell_characteristic_frequency / (maxwell_characteristic_frequency + omega * I)


def b_computing(
    maxwell_characteristic_frequency: Expr,
    kelvin_characteristic_frequency: Expr,
    burgers_characteristic_frequency: Expr,
    omega: Expr,
) -> Expr:
    """
    Computes b transfert function value given the Maxwell's, Kelvin's and Burgers characteristic
    frequencies and pulsation value omega.
    """

    return (omega * I * burgers_characteristic_frequency) / (
        (omega * I + kelvin_characteristic_frequency)
        * (omega * I + maxwell_characteristic_frequency)
    )


def unbounded_f_attenuation_computing(
    frequency: Expr, alpha: Expr, period_unit: Expr, omega_m: Expr
) -> Expr:
    """
    Computes the unbounded (short-term approximation) attenuation function f using parameters
    omega_m and alpha. Variables frequency and omega_m should represent unitless frequencies.
    """

    omega_0 = 1.0
    frequency_unit = 1 / period_unit

    return Piecewise(
        (2.0 / pi * log((frequency * frequency_unit) / omega_0) + 1.0j, frequency >= omega_m),
        (
            2.0 / (pi * alpha) * (1 - (omega_0 / (frequency * frequency_unit)) ** alpha)
            + 1j * (omega_0 / (frequency * frequency_unit)) ** alpha,
            True,
        ),
    )


def find_tau_M(omega_m: Expr, alpha: Expr, asymptotic_mu_ratio: Expr, q_mu: Expr) -> Expr:
    """
    Uses asymptotic equation to find tau_M such as it is constrained by the asymptotic mu ratio,
    which is equivalent to the relative amplitude of the transient response compared to the elastic
    response.
    """

    return (alpha * (1.0 - asymptotic_mu_ratio) * q_mu + omega_m ** (-alpha)) ** (1.0 / alpha)


def bounded_f_attenuation_computing(
    omega: Expr,
    alpha: Expr,
    omega_m: Expr,
    tau_M: Expr,
    period_unit: Expr,
) -> Expr:
    """
    Computes the bounded (full-domain valid) attenuation function f using parameters tau_m, tau_M
    and alpha. Variables tau_m and tau_M should represent unitless times and omega should represent
    a unitless pulsation.
    """

    tau_0 = 1.0
    frequency_unit = 1 / period_unit
    tau = Symbol("tau")
    tau_m = 2 * pi / omega_m

    return integrate(
        (tau / tau_0) ** alpha / (1 + I * omega * frequency_unit * tau) / tau,
        (tau, tau_m * period_unit, tau_M * period_unit),
    )


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


def create_rheological_expressions(bounded_attenuation_functions: bool = True) -> dict[str, Expr]:
    """
    Creates all the needed rheological expressions from scratch. Every terminal attribute has yet to
    be replaced by it spolynomial expression.
    """

    expressions: dict[str, Expr] = {
        str(symbol): symbol
        for symbol in symbols(
            r"omega rho_0 v_s alpha omega_m Delta q_mu T_{unit} eta_m mu_k1 c eta_k f_{attenuation}"
        )
    }
    expressions[r"mu_0"] = mu_0_computing(rho_0=expressions[r"rho_0"], v_s=expressions[r"v_s"])
    maxwell_characteristic_frequency = characteristic_frequency_computing(
        mu=expressions[r"mu_0"], eta=expressions[r"eta_m"]
    )
    m_prime = m_prime_computing(
        maxwell_characteristic_frequency=maxwell_characteristic_frequency,
        omega=expressions[r"omega"],
    )
    b = b_computing(
        maxwell_characteristic_frequency=maxwell_characteristic_frequency,
        kelvin_characteristic_frequency=characteristic_frequency_computing(
            mu=mu_k_computing(
                mu_k1=expressions[r"mu_k1"], c=expressions[r"c"], mu_0=expressions[r"mu_0"]
            ),
            eta=expressions[r"eta_k"],
        ),
        burgers_characteristic_frequency=characteristic_frequency_computing(
            mu=expressions[r"mu_0"], eta=expressions[r"eta_k"]
        ),
        omega=expressions[r"omega"],
    )
    expressions[r"mu_{complex}"] = mu_computing(
        mu_0=expressions[r"mu_0"],
        q_mu=expressions[r"q_mu"],
        f_attenuation=expressions[r"f_{attenuation}"],
        m_prime=m_prime,
        b=b,
    )

    return expressions


def attenuation_function_computing(
    expressions: dict[str, Expr], bounded_attenuation_functions: bool = True
) -> Expr:
    """
    Attenuation function formula, whether it is bounded or not.
    """

    return (
        unbounded_f_attenuation_computing(
            omega=expressions[r"omega"],
            alpha=expressions[r"alpha"],
            period_unit=expressions[r"T_{unit}"],
            omega_m=expressions[r"omega_m"],
        )
        if not bounded_attenuation_functions
        else bounded_f_attenuation_computing(
            omega=expressions[r"omega"],
            alpha=expressions[r"alpha"],
            omega_m=expressions[r"omega_m"],
            tau_M=find_tau_M(
                omega_m=expressions[r"omega_m"],
                alpha=expressions[r"alpha"],
                asymptotic_mu_ratio=expressions[r"Delta"],
                q_mu=expressions[r"q_mu"],
            ),
            period_unit=expressions[r"T_{unit}"],
        )
    )


def fluid_system_matrix(n: Expr, rho_0: Expr, g_0: Expr, x: Expr) -> MutableDenseMatrix:
    """
    Defines the fluid integration system matrix.
    """

    # Smylie (2013) Eq.9.42 & 9.43.
    c_1_1 = 4.0 * rho_0 / g_0

    return MutableDenseMatrix(
        [
            [c_1_1, One()],
            [(n * (n + 1.0) / x**2) - 16.0 * rho_0 / (g_0 * x), (-2.0 / x) - c_1_1],
        ]
    )


# TODO:
def solid_system_matrix(
    n: Expr,
    omega: Expr,
    lambda_complex: Expr,
    mu_complex: Expr,
    rho_0: Expr,
    g_0: Expr,
    x: Expr,
) -> None:
    """
    Defines the solid integration system matrix.
    """

    # Intermediate variables for lisibility.
    dyn_term = -rho_0 * omega**2.0  # TODO: Manage case and elastic.
    n_1 = n * (n + 1.0)
    b = 1.0 / (lambda_complex + 2.0 * mu_complex)
    c = 2.0 * mu_complex * (3.0 * lambda_complex + 2.0 * mu_complex) * b

    return MutableDenseMatrix(
        [
            [
                -2.0 * lambda_complex * b / x,
                b,
                n_1 * lambda_complex * b / x,
                Zero(),
                Zero(),
                Zero(),
            ],
            [
                (-4.0 * g_0 * rho_0 / x) + (2.0 * c / (x**2)) + dyn_term,
                -4.0 * mu_complex * b / x,
                n_1 * (rho_0 * g_0 / x - c / (x**2)),
                n_1 / x,
                Zero(),
                -rho_0,
            ],
            [-1.0 / x, Zero(), 1.0 / x, 1.0 / mu_complex, Zero(), Zero()],
            [
                rho_0 * g_0 / x - c / (x**2),
                -lambda_complex * b / x,
                (4.0 * n_1 * mu_complex * (lambda_complex + mu_complex) * b - 2.0 * mu_complex)
                / (x**2)
                + dyn_term,
                -3.0 / x,
                -rho_0 / x,
                Zero(),
            ],
            [4.0 * rho_0, Zero(), Zero(), Zero(), Zero(), One()],
            [Zero(), Zero(), -4.0 * rho_0 * n_1 / x, Zero(), n_1 / (x**2), -2.0 / x],
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
        m_vector = Matrix(g_matrix_inv @ d_matrix.T).flat()

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

    return love.reshape((3, 3))
