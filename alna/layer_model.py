"""
Solid Earth model description class for preprocessing.
"""

from typing import Optional

from numpy import inf
from sympy import Expr, Symbol

from .constants import LAYER_DECIMALS
from .parameters import DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS, IntegrationParameters


class LayerModel:
    """
    Describes what parameterizes a single layer.
    """

    r_inf: float = 0.0
    r_sup: float = 0.0
    name: str = ""
    polynomials: dict[str, list[float]] = {}
    parameter_symbols: dict[str, list[Expr]] = {}
    propagator: Optional[Expr] = None
    partial_propagators: Optional[dict[str, Expr]] = None

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
