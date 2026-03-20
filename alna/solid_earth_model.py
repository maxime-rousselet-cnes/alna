"""
Solid Earth model description class for preprocessing.
"""

from pathlib import Path
from typing import Optional

from base_models import (
    EARTH_RADIUS,
    SolidEarthModelPart,
    adaptive_runge_kutta_45,
    load_base_model,
    save_base_model,
)
from numpy import array, inf, ndarray
from sympy import Expr, Symbol, srepr

from .constants import SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH
from .parameters import (
    DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS,
    IntegrationParameters,
    SolidEarthParameters,
)


def high_degree_approximation(
    x: float,
    n: int,
    integration_parameters: IntegrationParameters = DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS,
) -> bool:
    """
    Checks whether to use high degrees approximation or not.
    """

    return x**n < integration_parameters.high_degrees_radius_sensibility


class LayerModel:
    """
    Describes what parameterizes a single layer.
    """

    x_inf: float = 0.0
    x_sup: float = 0.0
    name: str = ""
    polynomials: dict[str, list[float]] = {}
    parameter_symbols: dict[str, list[Expr]] = {}

    def __init__(
        self,
        r_inf: float,
        r_sup: float,
        name: Optional[str] = None,
        earth_radius: float = EARTH_RADIUS,
    ) -> None:

        self.x_inf = r_inf / earth_radius
        self.x_sup = r_sup / earth_radius

        if name:

            self.name = name

    def update_polynomials(self, polynomials: dict[str, list[list[float]]], i_layer: int) -> None:
        """
        Updates the layer data structure attributes with input numerical values and create the
        expressions accordingly.
        """

        self.polynomials = {
            parameter: polynomials_per_layer[i_layer]
            for parameter, polynomials_per_layer in polynomials.items()
        }
        self.parameter_symbols = {
            parameter: [
                Symbol("_".join((parameter, self.name, str(k))))
                for k in range(len(polynomials_per_layer[i_layer]))
            ]
            for parameter, polynomials_per_layer in polynomials.items()
        }

    def integrate_y_i_system(
        self,
        y_i: ndarray,
        n: int,
        integration_parameters: IntegrationParameters = DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS,
    ) -> tuple[ndarray, ndarray]:
        """
        Integrates the y_i system from the bottom to the top of the layer.
        """

        # Numerical approximation for high degrees.
        if high_degree_approximation(
            x=self.x_sup, n=n, integration_parameters=integration_parameters
        ):
            return array(object=[self.x_inf]), array(object=[y_i])

        # TODO: RK45.
        return array(object=[self.x_inf]), array(object=[y_i])

    def to_serializable(
        self,
    ) -> dict[str, float | str | dict[str, list[float | str]] | dict[str, list[str]]]:
        """
        To facilitate (.JSON) saving and loading.
        """

        return {
            "x_inf": self.x_inf,
            "x_sup": self.x_sup,
            "name": self.name,
            "polynomials": {
                parameter: ["inf"] if inf in values else values
                for parameter, values in self.polynomials.items()
            },
            "parameter_symbols": {
                parameter: [srepr(expression) for expression in expressions]
                for parameter, expressions in self.parameter_symbols.items()
            },
        }

    def get_parameters_dict(self) -> tuple[dict[str, float], dict[str, Expr]]:
        """
        Returns the dictionary of parameter values and the dictionary of parameter expressions.
        """

        parameter_values = {}
        parameter_expresisons = {}

        for parameter, polynomial in self.polynomials.items():

            parameter_values |= {
                str(self.parameter_symbols[parameter][polynomial_degree]): polynomial_coefficient
                for polynomial_degree, polynomial_coefficient in enumerate(polynomial)
            }
            parameter_values |= {
                str(self.parameter_symbols[parameter][polynomial_degree]): self.parameter_symbols[
                    parameter
                ][polynomial_degree]
                for polynomial_degree in range(len(polynomial))
            }

        return parameter_values, parameter_expresisons


class SolidEarthNumericalModel:

    layer_models: list[LayerModel] = []

    # TODO.
    def save(self) -> None:
        """
        TODO.
        """

        pass


# TODO: Load. Merge: new instance from scratch.


class SolidEarthModelDescription:
    """
    Describes physical quantities by polynomials depending on the unitless radius.
    Can be used to encode all different parts of a planetary rheology.
    """

    # Names of the spherical layers.
    layer_names: list[Optional[str]]
    # Boundaries of the spherical layers.
    r_limits: list[float]
    # Optionally changes the crust values by a constant. Usefull for continental/oceanic crust.
    optional_crust_values: dict[str, Optional[float]]

    # Polynomials (depending on x := unitless r) of physical quantities describing the planetary
    # model. The keys are the
    # variable names. They should include:
    #   - for the elastic part:
    #       - v_s: S wave velocity (m.s^-1).
    #       - v_p: P wave velocity (m.s^-1).
    #       - rho_0: Density (kg.m^-3).
    #   - for the attenuation part:
    #       - q_mu: Shear modulus attenuation coefficient Q (unitless).
    #   - for the transient part:
    #       - omega_m: (Hz).
    #       - alpha: (Unitless).
    #       - asymptotic_mu_ratio: Defines mu(omega -> 0.0) / mu_0 (Unitless).
    #   - for the viscous part:
    #       - eta_m: Maxwell's viscosity (Pa.s).
    #       - (Optional) eta_k: Kelvin's viscosity (Pa.s).
    #       - (Optional) mu_k1: Kelvin's elasticity constant term (Pa).
    #       - (Optional) c: Elasticities ratio, such as mu_K = c * mu_E + mu_k1 (Unitless).
    polynomials: dict[str, list[list[float | str]]]

    def __init__(
        self,
        name: str,
        solid_earth_model_part: Optional[SolidEarthModelPart] = None,
        path: Optional[Path] = None,
    ) -> None:
        """
        Loads the model file while managing infinite values.
        """

        if solid_earth_model_part is None and path is None:

            raise NotImplementedError

        loaded_content = load_base_model(
            name=name,
            path=(
                path
                if not path is None
                else SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH[solid_earth_model_part.value]
            ),
        )
        self.layer_names = loaded_content["layer_names"]
        self.r_limits = loaded_content["r_limits"]
        self.optional_crust_values = loaded_content["optional_crust_values"]
        self.polynomials = loaded_content["polynomials"]
        self.manage_infinite_cases()

    def manage_infinite_cases(self) -> None:
        """
        To set infinite numpy values in data structure whereas "inf" string is lisible in the
        (.JSON) files.
        """

        for parameter, polynomials_per_layer in self.polynomials.items():

            for i_layer, polynomial in enumerate(polynomials_per_layer):

                if "inf" in polynomial:

                    self.polynomials[parameter][i_layer] = [inf]

    def save(self, name: str, path: Path):
        """
        Saves in (.JSON) file. Replace back infinite values by strings.
        """

        for parameter, polynomials_per_layer in self.polynomials.items():

            for i_layer, polynomial in enumerate(polynomials_per_layer):

                if inf in polynomial:

                    self.polynomials[parameter][i_layer] = ["inf"]

        save_base_model(obj=self.__dict__, name=name, path=path)
        self.manage_infinite_cases()

    def generate_solid_earth_numerical_model(
        self,
        solid_earth_parameters: SolidEarthParameters = DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS,
    ) -> SolidEarthNumericalModel:
        """ """

        layer_models: list[LayerModel] = []

        for i_layer, layer_name in enumerate(self.layer_names):

            layer_models += [
                LayerModel(
                    r_inf=self.r_limits[i_layer],
                    r_sup=self.r_limits[i_layer + 1],
                    name=layer_name,
                    earth_radius=solid_earth_parameters.model.radius_unit,
                )
            ]

            layer_models[-1].update_polynomials(polynomials=self.polynomials, i_layer=i_layer)

        return SolidEarthNumericalModel(layer_models=layer_models)
