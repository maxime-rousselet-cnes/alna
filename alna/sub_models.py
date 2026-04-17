"""
Solid Earth model description class for preprocessing.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from base_models import evaluate_terminal_parameters
from numpy import inf, ndarray
from sympy import Expr, Matrix, Symbol
from sympy.core.numbers import Zero

from .constants import LAYER_DECIMALS, Y_I_STATE_FOR_SURFACE, Y_I_STATE_VECTOR_LINE
from .parameters import (
    DEFAULT_COMPONENT_PARAMETERS,
    DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS,
    IntegrationParameters,
    SolidEarthModelParameters,
)
from .rheological_formulas import (
    create_rheological_expressions,
    fluid_system_matrix,
    g_0_computing,
    solid_system_matrix,
    surface_solution,
)


class LayerModel:
    """
    Describes what parameterizes a single layer.
    """

    r_inf: float = 0.0
    r_sup: float = 0.0
    name: str = ""
    polynomials: dict[str, list[float]] = {}
    parameter_symbols: dict[str, list[Expr]] = {}
    propagator: Optional[Callable | Expr] = None
    partial_propagators: Optional[dict[str, Callable]] = None

    def __init__(self, r_inf: float = 0.0, r_sup: float = 0.0, name: str = ""):

        self.r_inf = round(number=r_inf, ndigits=LAYER_DECIMALS)
        self.r_sup = round(number=r_sup, ndigits=LAYER_DECIMALS)
        self.name = name

    def update_polynomials(self, polynomials: dict[str, list[list[float]]], i_layer: int) -> None:
        """
        Updates the layer data structure attributes with input numerical values and create the
        expressions accordingly.
        """

        if self.polynomials == {}:

            self.polynomials = {}
            self.parameter_symbols = {}

        self.polynomials |= {
            parameter: polynomials_per_layer[i_layer]
            for parameter, polynomials_per_layer in polynomials.items()
        }

        self.parameter_symbols |= {
            parameter: [
                Symbol(parameter + "^{" + "_".join((self.name, str(k))) + "}")
                for k in range(len(polynomials_per_layer[i_layer]))
            ]
            for parameter, polynomials_per_layer in polynomials.items()
        }

    def to_serializable(
        self,
    ) -> dict[str, float | str | dict[str, list[float | str]] | dict[str, list[str]]]:
        """
        To facilitate (.JSON) saving and loading.
        """

        return {
            "r_inf": self.r_inf,
            "r_sup": self.r_sup,
            "name": self.name,
            "polynomials": {
                parameter: ["inf"] if inf in values else values
                for parameter, values in self.polynomials.items()
            },
            "parameter_symbols": {
                parameter: [str(expression) for expression in expressions]
                for parameter, expressions in self.parameter_symbols.items()
            },
        }

    def get_parameters_dict(self) -> tuple[dict[str, float], dict[str, Expr]]:
        """
        Returns the dictionary of parameter values and the dictionary of parameter expressions.
        """

        parameter_values = {}
        parameter_expressions = {}

        for parameter, polynomial in self.polynomials.items():

            parameter_values |= {
                str(self.parameter_symbols[parameter][polynomial_degree]): polynomial_coefficient
                for polynomial_degree, polynomial_coefficient in enumerate(polynomial)
            }
            parameter_expressions |= {
                str(self.parameter_symbols[parameter][polynomial_degree]): self.parameter_symbols[
                    parameter
                ][polynomial_degree]
                for polynomial_degree in range(len(polynomial))
            }

        return parameter_values, parameter_expressions

    def evaluate(
        self, radius_unit: float, variable: float = r"\rho_0", r_inf: bool = False
    ) -> float:
        """
        Evaluates the given variable either in r = r_inf or r = r_sup.
        Beware the variable still eventually has a unit.
        """

        return sum(
            coefficient * ((self.r_inf if r_inf else self.r_sup) / radius_unit) ** degree
            for degree, coefficient in enumerate(self.polynomials[variable])
        )

    def high_degree_approximation(
        self,
        radius_unit: float,
        n: int,
        integration_parameters: IntegrationParameters = DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS,
    ) -> bool:
        """
        Checks whether to use high degrees approximation or not.
        """

        return (
            self.r_sup / radius_unit
        ) ** n < integration_parameters.high_degrees_radius_sensibility


class Expressions:
    """
    All sympy related attributes needed by a solid Earth numerical model.
    """

    expressions: dict[str, Expr] = {}
    parameter_expressions: dict[str, Expr] = {}
    terminal_parameter_values: dict[str, float] = {}

    def evaluate(
        self,
        expression: Expr | str,
        x: Optional[float] = None,
    ) -> Expr:
        """
        Replaces all parameter expressions in a given expression by their numerical values.
        """

        evaluated_expression = evaluate_terminal_parameters(
            expression=(
                expression if not isinstance(expression, str) else self.expressions[expression]
            ),
            parameter_expressions=self.parameter_expressions,
            terminal_parameter_values=self.terminal_parameter_values,
        )

        return (
            evaluated_expression
            if x is None
            else evaluated_expression.xreplace(rule={self.expressions[r"x"]: x})
        )

    def create_propagators(
        self,
        model: SolidEarthModelParameters,
        layer_models: list[LayerModel],
        units: dict[str, float],
    ) -> None:
        """
        Creates all rheological expressions, for every layer, needed for the y_i system
        integration.
        """

        self.expressions[r"g_0"] = Zero()
        self.expressions[r"x"] = Symbol(r"x")
        self.expressions[r"n"] = Symbol(r"n")
        self.expressions[r"\omega"] = Symbol(r"\omega")

        for i_layer, layer_model in enumerate(layer_models):

            x_inf = layer_model.r_inf / model.radius_unit
            self.expressions |= {
                parameter: sum(
                    (
                        # Makes all parameters unitless.
                        (symbol / (1.0 if parameter not in units else units[parameter]))
                        * self.expressions[r"x"] ** degree
                        for degree, symbol in enumerate(polynomial_expression)
                    ),
                    start=Zero(),
                )
                for parameter, polynomial_expression in layer_model.parameter_symbols.items()
            }
            self.expressions = create_rheological_expressions(
                expressions=self.expressions,
                units=units,
                component_parameters=(
                    DEFAULT_COMPONENT_PARAMETERS
                    if i_layer < model.structure_parameters.i_layer_cmb
                    else model.component_parameters
                ),
            )
            self.expressions[r"g_0"] = g_0_computing(
                x=self.expressions[r"x"],
                rho_0=self.expressions[r"\rho_0"],
                g_0_inf=self.expressions[r"g_0"],
                x_inf=x_inf,
            )
            self.expressions[r"g_0^{layer_{" + str(i_layer) + "}}"] = self.expressions[r"g_0"]
            matrix_expression = (
                fluid_system_matrix(expressions=self.expressions)
                if model.structure_parameters.i_layer_cmb
                > i_layer
                >= model.structure_parameters.i_layer_icb
                else solid_system_matrix(
                    expressions=self.expressions,
                    components=(
                        DEFAULT_COMPONENT_PARAMETERS
                        if i_layer < model.structure_parameters.i_layer_cmb
                        else model.component_parameters
                    ),
                )
            )
            layer_models[i_layer].propagator = Matrix(
                matrix_expression
                @ Matrix(
                    Y_I_STATE_VECTOR_LINE[
                        : (
                            2
                            if (
                                model.structure_parameters.i_layer_cmb
                                > i_layer
                                >= model.structure_parameters.i_layer_icb
                            )
                            else 6
                        )
                    ]
                )
            )
            terminal_parameter_values, parameter_expressions = layer_models[
                i_layer
            ].get_parameters_dict()
            self.terminal_parameter_values |= terminal_parameter_values
            self.parameter_expressions |= parameter_expressions

    def define_love_number_expressions(self, n: int) -> None:
        """
        (3, 3) Love numbers from (3, 6) y_i surface solutions.
        """

        self.expressions[r"L_n"] = surface_solution(
            n=n,
            y_1_s=Y_I_STATE_FOR_SURFACE[0],
            y_2_s=Y_I_STATE_FOR_SURFACE[1],
            y_3_s=Y_I_STATE_FOR_SURFACE[2],
            g_0_surface=self.expressions[r"g_0"],
        )


@dataclass
class IntegrationContext:
    """
    Describes the local context of an already performed integration. Needed for partial derivatives
    quadrature.
    """

    n: int
    i_omega: int
    omega: float
    x_tabs: list[list[ndarray]]
    y_tabs: list[list[ndarray]]
