"""
Solid Earth model description class for preprocessing.
"""

from pathlib import Path
from typing import Optional

from base_models import (
    SolidEarthModelPart,
    adaptive_runge_kutta_45,
    evaluate_terminal_parameters,
    load_base_model,
    non_adaptive_runge_kutta_45,
    partial_symbols,
    save_base_model,
    vector_variation_equation,
)
from mpmath import exp
from numpy import array, empty, inf, ndarray, pi, zeros
from pydantic import BaseModel, ConfigDict
from sympy import Expr, Matrix, Symbol, flatten, lambdify
from sympy.core.numbers import Zero

from .constants import (
    INITIAL_Y_I,
    LAYERS_SEPARATOR,
    SECONDS_PER_YEAR,
    SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
    Y_I_STATE_FOR_SURFACE,
    Y_I_STATE_VECTOR_LINE,
    G,
)
from .layer_model import LayerModel
from .parameters import DEFAULT_COMPONENT_PARAMETERS, SolidEarthParameters
from .rheological_formulas import (
    create_rheological_expressions,
    fluid_system_matrix,
    fluid_to_solid,
    g_0_computing,
    solid_system_matrix,
    solid_to_fluid,
    surface_solution,
)


class SolidEarthNumericalModel(BaseModel):
    """
    Describes a solid Earth model numerically. Manages the pre-processing when instanciating and
    merging. Manages numerical integrations as methods.
    """

    name: str
    layer_models: list[LayerModel]
    solid_earth_parameters: SolidEarthParameters
    units: dict[str, float] = {}
    love_numbers: dict[float, ndarray] = {}
    love_number_partials: dict[str, dict[float, ndarray]] = {}
    # Unsaved attributes.
    expressions: dict[str, Expr] = {}
    parameter_expressions: dict[str, Expr] = {}
    terminal_parameter_values: dict[str, float] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)  # To authorize arrays.

    def save(self, path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH) -> None:
        """
        Serialize the numerical model and saves in a (.JSON) file.
        """

        save_base_model(
            obj={
                "name": self.name,
                "layer_models": [
                    layer_model.to_serializable() for layer_model in self.layer_models
                ],
                # Does not save expresisons.
                "solid_earth_parameters": self.solid_earth_parameters,
                "units": self.units,
                "love_numbers": self.love_numbers,
                "love_number_partials": self.love_number_partials,
            },
            name=self.name,
            path=path,
        )

    def merge(
        self,
        solid_earth_model_description: "SolidEarthModelDescription",
        name: str,
    ) -> None:
        """
        Merges with another component
        """

        self.name = SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join((self.name, name))
        r_inf = self.layer_models[
            self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
        ].r_inf
        i_layer_main = self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
        i_layer_merging_component = 0
        new_layer_models = self.layer_models[
            : self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
        ]
        merging_component_layer_models = [
            LayerModel(r_inf=r_inf, r_sup=r_sup, name=layer_name)
            for r_inf, r_sup, layer_name in zip(
                solid_earth_model_description.r_limits[:-1],
                solid_earth_model_description.r_limits[1:],
                solid_earth_model_description.layer_names,
            )
        ]

        for i_layer, layer_model in enumerate(merging_component_layer_models):

            layer_model.update_polynomials(
                polynomials=solid_earth_model_description.polynomials, i_layer=i_layer
            )

        while r_inf < self.solid_earth_parameters.model.radius_unit:

            memorized_r_inf = r_inf
            layer_name = LAYERS_SEPARATOR.join(
                (
                    self.layer_models[i_layer_main].name,
                    solid_earth_model_description.layer_names[i_layer_merging_component],
                )
            )
            polynomials = (
                self.layer_models[i_layer_main].polynomials
                | merging_component_layer_models[i_layer_merging_component].polynomials
            )
            parameter_symbols = (
                self.layer_models[i_layer_main].parameter_symbols
                | merging_component_layer_models[i_layer_merging_component].parameter_symbols
            )

            if (
                solid_earth_model_description.r_limits[i_layer_merging_component + 1]
                < self.layer_models[i_layer_main].r_sup
            ):

                i_layer_merging_component += 1
                r_inf = solid_earth_model_description.r_limits[i_layer_merging_component]

            else:

                r_inf = self.layer_models[i_layer_main].r_sup
                i_layer_main += 1

            if r_inf != memorized_r_inf:

                new_layer_models += [
                    LayerModel(r_inf=memorized_r_inf, r_sup=r_inf, name=layer_name)
                ]
                new_layer_models[-1].polynomials = polynomials
                new_layer_models[-1].parameter_symbols = parameter_symbols

        self.layer_models = new_layer_models

    def create_propagators(self) -> None:
        """
        Generates all needed symbols for the Y_i system symbolic definition.
        """

        self.units = {
            r"R": self.solid_earth_parameters.model.radius_unit,
            r"\rho_0": self.layer_models[
                self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
            ].evaluate(
                radius_unit=self.solid_earth_parameters.model.radius_unit,
                r_inf=True,
            ),
        }
        self.units[r"T"] = (self.units[r"\rho_0"] * pi * G) ** (-0.5)
        self.units[r"\eta_m"] = self.units[r"\rho_0"] * self.units[r"R"] ** 2 / self.units[r"T"]
        self.units[r"v_p"] = self.units[r"R"] / self.units[r"T"]
        self.units[r"g_0"] = self.units[r"R"] / self.units[r"T"] ** 2
        self.units[r"\mu_0"] = self.units[r"\rho_0"] * self.units[r"R"] ** 2 / self.units[r"T"] ** 2
        self.units[r"v_s"] = self.units[r"v_p"]
        self.units[r"\lambda_0"] = self.units[r"\mu_0"]
        self.units[r"f"] = 1.0 / self.units[r"T"]
        self.units[r"\omega_m^{inf}"] = self.units[r"f"]
        self.units[r"\mu_{k1}"] = self.units[r"\mu_0"]
        self.units[r"\eta_k"] = self.units[r"\eta_m"]
        self.expressions[r"g_0"] = Zero()
        self.expressions[r"x"] = Symbol(r"x")
        self.expressions[r"n"] = Symbol(r"n")
        self.expressions[r"\omega"] = Symbol(r"\omega")
        i_layer_icb = self.solid_earth_parameters.model.structure_parameters.i_layer_icb

        for i_layer, layer_model in enumerate(self.layer_models):

            x_inf = layer_model.r_inf / self.solid_earth_parameters.model.radius_unit
            self.expressions |= {
                parameter: sum(
                    (
                        # Makes all parameters unitless.
                        (symbol / (1.0 if parameter not in self.units else self.units[parameter]))
                        * self.expressions[r"x"] ** degree
                        for degree, symbol in enumerate(polynomial_expression)
                    ),
                    start=Zero(),
                )
                for parameter, polynomial_expression in layer_model.parameter_symbols.items()
            }
            self.expressions = create_rheological_expressions(
                expressions=self.expressions,
                units=self.units,
                component_parameters=(
                    DEFAULT_COMPONENT_PARAMETERS
                    if i_layer < self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
                    else self.solid_earth_parameters.model.component_parameters
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
                if self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
                > i_layer
                >= self.solid_earth_parameters.model.structure_parameters.i_layer_icb
                else solid_system_matrix(
                    expressions=self.expressions,
                    components=(
                        DEFAULT_COMPONENT_PARAMETERS
                        if i_layer
                        < self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
                        else self.solid_earth_parameters.model.component_parameters
                    ),
                )
            )
            self.layer_models[i_layer].propagator = Matrix(
                matrix_expression
                @ Matrix(
                    Y_I_STATE_VECTOR_LINE[
                        : (
                            2
                            if (
                                self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
                                > i_layer
                                >= i_layer_icb
                            )
                            else 6
                        )
                    ]
                )
            )
            terminal_parameter_values, parameter_expressions = self.layer_models[
                i_layer
            ].get_parameters_dict()
            self.terminal_parameter_values |= terminal_parameter_values
            self.parameter_expressions |= parameter_expressions

    def integrate_y_i_layer(
        self, i_layer: int, y_i: ndarray, n: int, propagator: Expr
    ) -> tuple[ndarray, ndarray]:
        """
        Integrates the y_i system through a layer.
        """

        if self.layer_models[i_layer].high_degree_approximation(
            radius_unit=self.solid_earth_parameters.model.radius_unit,
            n=n,
            integration_parameters=self.solid_earth_parameters.integration_parameters,
        ):

            return array(object=[self.layer_models[i_layer].r_inf]), array(object=[y_i])

        i_layer_icb = self.solid_earth_parameters.model.structure_parameters.i_layer_icb
        x, y = adaptive_runge_kutta_45(
            fun=lambdify(
                args=[
                    self.expressions[r"x"],
                    Y_I_STATE_VECTOR_LINE[
                        : (
                            2
                            if (
                                self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
                                > i_layer
                                >= i_layer_icb
                            )
                            else 6
                        )
                    ],
                ],
                expr=flatten(propagator),
                modules=(
                    [{"exp_polar": exp}, "mpmath"]
                    if i_layer >= self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
                    and self.solid_earth_parameters.model.component_parameters.transient_component
                    else "numpy"
                ),
            ),
            t_bounds=(
                max(
                    self.layer_models[i_layer].r_inf,
                    self.solid_earth_parameters.integration_parameters.minimal_radius,
                )
                / self.solid_earth_parameters.model.radius_unit,
                self.layer_models[i_layer].r_sup / self.solid_earth_parameters.model.radius_unit,
                inf,
            ),
            y_0=y_i,
        )

        return x, y

    def integrate_y_i_system(
        self,
        n: int,
        omega: float,
    ) -> tuple[list[list[ndarray]], list[list[ndarray]]]:
        """
        Performs the y_i system integration.
        """

        y_1, y_2, y_3 = INITIAL_Y_I

        # Per layer and per boundary condition.
        x_tabs: list[list[ndarray]] = []
        y_tabs: list[list[ndarray]] = []

        for i_layer, layer_model in enumerate(self.layer_models):

            propagator = layer_model.propagator.xreplace(
                rule={
                    self.expressions[r"n"]: n,
                    self.expressions[r"\omega"]: omega / self.units[r"f"],
                }
            )

            # Inner core layers.
            if i_layer < self.solid_earth_parameters.model.structure_parameters.i_layer_icb:

                # The whole x-dependent solution is not needed for the core.
                y_1 = self.integrate_y_i_layer(
                    i_layer=i_layer, y_i=y_1, n=n, propagator=propagator
                )[1][-1]
                y_2 = self.integrate_y_i_layer(
                    i_layer=i_layer, y_i=y_2, n=n, propagator=propagator
                )[1][-1]
                y_3 = self.integrate_y_i_layer(
                    i_layer=i_layer, y_i=y_3, n=n, propagator=propagator
                )[1][-1]

            # Fluid core layers.
            elif i_layer < self.solid_earth_parameters.model.structure_parameters.i_layer_cmb:

                if i_layer == self.solid_earth_parameters.model.structure_parameters.i_layer_icb:

                    # Meaning integration has been skipped in the Inner core for high degrees.
                    if y_3[3] == 0.0:

                        y_fluid = array(object=[0.0, 1.0], dtype=complex)

                    else:

                        y_fluid = solid_to_fluid(
                            y_1=y_1,
                            y_2=y_2,
                            y_3=y_3,
                            rho_0_fluid_inf=layer_model.evaluate(
                                radius_unit=self.solid_earth_parameters.model.radius_unit,
                                r_inf=True,
                            )
                            / self.units[r"\rho_0"],
                            g_0_fluid_inf=evaluate_terminal_parameters(
                                expression=self.expressions[r"g_0^{layer_{" + str(i_layer) + "}}"],
                                parameter_expressions=self.parameter_expressions,
                                terminal_parameter_values=self.terminal_parameter_values,
                            ).xreplace(
                                rule={
                                    self.expressions[r"x"]: layer_model.r_inf
                                    / self.solid_earth_parameters.model.radius_unit,
                                }
                            ),
                        )

                # The whole x-dependent solution is not needed for the core.
                y_fluid = self.integrate_y_i_layer(
                    i_layer=i_layer, y_i=y_fluid, n=n, propagator=propagator
                )[1][-1]

                if (
                    i_layer
                    == self.solid_earth_parameters.model.structure_parameters.i_layer_cmb - 1
                ):

                    y_init = fluid_to_solid(
                        yf_1=y_fluid,
                        rho_0_fluid_sup=layer_model.evaluate(
                            radius_unit=self.solid_earth_parameters.model.radius_unit,
                        )
                        / self.units[r"\rho_0"],
                        g_0_fluid_sup=evaluate_terminal_parameters(
                            expression=self.expressions[r"g_0^{layer_{" + str(i_layer) + "}}"],
                            parameter_expressions=self.parameter_expressions,
                            terminal_parameter_values=self.terminal_parameter_values,
                        ).xreplace(
                            rule={
                                self.expressions[r"x"]: layer_model.r_sup
                                / self.solid_earth_parameters.model.radius_unit,
                            }
                        ),
                    )

            # Mantle and crust layers.
            else:

                x_tabs += [[empty(shape=()), empty(shape=()), empty(shape=())]]
                y_tabs += [[empty(shape=()), empty(shape=()), empty(shape=())]]

                for i, y_i in enumerate(y_init):

                    x_tabs[-1][i], y_tabs[-1][i] = self.integrate_y_i_layer(
                        i_layer=i_layer, y_i=y_i, n=n, propagator=propagator
                    )

                y_init = tuple(y_i_tab[-1] for y_i_tab in y_tabs[-1])

        return x_tabs, y_tabs

    def compute_love_numbers(
        self,
        period_tab_per_degree: dict[int, ndarray],
        parameters_to_invert: Optional[list[str]] = None,
    ) -> None:
        """
        Performs the y_i system integration for every degree and period of the list.
        """

        if parameters_to_invert is None:

            parameters_to_invert = []

        # Creates the y_i abstract system and updates all symbols and numerical values.
        self.create_propagators()
        partial_expressions_per_parameter, partials_matrix_per_parameter = {}, {}

        # Defines formally the partial derivative symbols.
        for parameter in parameters_to_invert:

            partial_expressions, partials_matrix_for_parameter = partial_symbols(
                parameter=parameter, state_vector_line=Y_I_STATE_VECTOR_LINE
            )
            partial_expressions_per_parameter[parameter] = partial_expressions
            partials_matrix_per_parameter[parameter] = partials_matrix_for_parameter

        # Applies the variation equations on the y_i system for every invertible parameter and
        # replaces all variables by their numerical values in the expressions. Only remain x, omega,
        # n, the 6 y_i and their partial derivatives.
        for i_layer, layer_model in enumerate(self.layer_models):

            for parameter in parameters_to_invert:

                layer_model.partial_propagators[parameter] = evaluate_terminal_parameters(
                    expression=vector_variation_equation(
                        dynamic=layer_model.propagator,
                        parameter=parameter,
                        partials=partials_matrix_per_parameter[parameter],
                        state_vector_line=Y_I_STATE_VECTOR_LINE,
                    ),
                    parameter_expressions=self.parameter_expressions,
                    terminal_parameter_values=self.terminal_parameter_values,
                )

            layer_model.propagator = evaluate_terminal_parameters(
                expression=layer_model.propagator,
                parameter_expressions=self.parameter_expressions,
                terminal_parameter_values=self.terminal_parameter_values,
            )

        self.love_numbers = {
            n: zeros(shape=(len(period_tab), 3, 3))
            for n, period_tab in period_tab_per_degree.items()
        }
        self.love_number_partials = {
            parameter: {
                n: zeros(shape=(len(period_tab), 3, 3))
                for n, period_tab in period_tab_per_degree.items()
            }
            for parameter in parameters_to_invert
        }

        for n, period_tab in period_tab_per_degree.items():

            omega_tab = 2 * pi / (SECONDS_PER_YEAR * period_tab)
            # Generates the Love number expression from y_i at x = 1. Depends numerically on n.
            love_number_expressions_from_surface_solutions = surface_solution(
                n=n,
                y_1_s=Y_I_STATE_FOR_SURFACE[0],
                y_2_s=Y_I_STATE_FOR_SURFACE[1],
                y_3_s=Y_I_STATE_FOR_SURFACE[2],
                g_0_surface=self.expressions[r"g_0"],
            )
            # Differentiates with respect to the parameters to invert.
            love_number_partial_expressions_from_surface_solutions = Zero()
            # Apply numerical values so that only the (3, 6) y_i remain.
            love_number_expressions_from_surface_solutions = evaluate_terminal_parameters(
                expression=love_number_expressions_from_surface_solutions,
                parameter_expressions=self.parameter_expressions,
                terminal_parameter_values=self.terminal_parameter_values,
            ).xreplace(rule={self.expressions[r"x"]: 1})

            for i_omega, omega in enumerate(omega_tab):

                # Integrates the raw y_i system (no partials) through every layer for the 3 boundary
                # conditions.
                x_tabs, y_tabs = self.integrate_y_i_system(n=n, omega=omega)

                for y_i_symbols, y_i_tab in zip(Y_I_STATE_FOR_SURFACE, y_tabs[-1]):

                    # Applies the expression of the solution to the (3, 6) y_i.
                    love_number_expressions_from_surface_solutions: Matrix = (
                        love_number_expressions_from_surface_solutions.xreplace(
                            rule=dict(zip(y_i_symbols, y_i_tab[-1]))
                        )
                    )

                self.love_numbers[n][i_omega] = array(
                    object=love_number_expressions_from_surface_solutions.doit()
                )

                for parameter in parameters_to_invert:

                    y_i_partials = zeros(shape=(3, 6))

                    # So layer_model actually represents the (i_layer + i_layer_cmb)-th layer.
                    for i_layer, layer_model in enumerate(
                        self.layer_models[
                            self.solid_earth_parameters.model.structure_parameters.i_layer_cmb :
                        ]
                    ):

                        partial_propagator = layer_model.partial_propagators[parameter].xreplace(
                            rule={
                                self.expressions[r"n"]: n,
                                self.expressions[r"\omega"]: omega / self.units[r"f"],
                            }
                        )

                        for i_boundary_condition, (x_tab, y_tab) in enumerate(
                            zip(x_tabs[i_layer], y_tabs[i_layer])
                        ):

                            y_i_partials[i_boundary_condition] = non_adaptive_runge_kutta_45(
                                fun=lambdify(
                                    args=[
                                        self.expressions[r"x"],
                                        Y_I_STATE_VECTOR_LINE,
                                        partial_expressions_per_parameter[parameter],
                                    ],
                                    expr=flatten(partial_propagator),
                                    modules=[{"exp_polar": exp}, "mpmath"],
                                ),
                                t=x_tab,
                                y_0=y_i_partials[i_boundary_condition],
                                parameters=y_tab,
                            )[
                                -1
                            ]  # The x-dependent partial is useless.

                    # Gets the surface solution partials from y_i_partials
                    self.love_number_partials[parameter][n][i_omega] = zeros(shape=(3, 3))


def load_solid_earth_numerical_model(
    name: str,
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
) -> SolidEarthNumericalModel:
    """
    Loads a solid Earth numerical model and formats its expressions.
    """

    loaded_content = load_base_model(name=name, path=path)
    love_numbers: dict[int, list[list[list[float]]]] = loaded_content["love_numbers"]
    love_number_partials: dict[str, dict[int, list[list[list[float]]]]] = loaded_content[
        "love_number_partials"
    ]
    solid_earth_numerical_model = SolidEarthNumericalModel(
        name=loaded_content["name"],
        layer_models=[LayerModel() for _ in range(len(loaded_content["layer_models"]))],
        solid_earth_parameters=loaded_content["solid_earth_parameters"],
        units=loaded_content["units"],
        love_numbers={
            n: array(object=love_numbers_tab) for n, love_numbers_tab in love_numbers.items()
        },
        love_number_partials={
            parameter: {
                n: array(object=love_number_partials_tab)
                for n, love_number_partials_tab in partials.items()
            }
            for parameter, partials in love_number_partials.items()
        },
    )
    layer_model: dict[str, dict | str | float]

    for i_layer, layer_model in enumerate(loaded_content["layer_models"]):

        solid_earth_numerical_model.layer_models[i_layer].name = layer_model["name"]
        solid_earth_numerical_model.layer_models[i_layer].r_inf = layer_model["r_inf"]
        solid_earth_numerical_model.layer_models[i_layer].r_sup = layer_model["r_sup"]
        solid_earth_numerical_model.layer_models[i_layer].polynomials = {
            parameter: [inf] if "inf" in polynomial else polynomial
            for parameter, polynomial in layer_model["polynomials"].items()
        }
        solid_earth_numerical_model.layer_models[i_layer].parameter_symbols = {
            quantity: [Symbol(parameter) for parameter in polynomial]
            for quantity, polynomial in layer_model["parameter_symbols"].items()
        }

    return solid_earth_numerical_model


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
    #       - \rho_0: Density (kg.m^-3).
    #   - for the attenuation part:
    #       - q_\mu: Shear modulus attenuation coefficient Q (unitless).
    #   - for the transient part:
    #       - \omega_m^{inf}: (Hz).
    #       - \alpha: (Unitless).
    #       - \Delta: Defines mu(omega -> 0.0) / mu_0 (Unitless).
    #   - for the viscous part:
    #       - \eta_m: Maxwell's viscosity (Pa.s).
    #       - (Optional) \eta_k: Kelvin's viscosity (Pa.s).
    #       - (Optional) \mu_{k1}: Kelvin's elasticity constant term (Pa).
    #       - (Optional) c: Elasticities ratio, such as mu_K = c * mu_E + mu_{k1} (Unitless).
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
        name: str,
        solid_earth_parameters: SolidEarthParameters,
    ) -> SolidEarthNumericalModel:
        """
        Pre-processes the profiles and create the needed elastic symbols for integration.
        """

        # Only allowed for elastic component.
        if not (
            "v_s" in self.polynomials
            and "v_p" in self.polynomials
            and r"\rho_0" in self.polynomials
        ):

            raise NotImplementedError

        layer_models: list[LayerModel] = []

        if solid_earth_parameters.model.optional_crust_values:

            for parameter in self.polynomials:

                self.polynomials[parameter][-1] = [self.optional_crust_values[parameter]]

        for i_layer, layer_name in enumerate(self.layer_names):

            layer_model = LayerModel(
                r_inf=self.r_limits[i_layer],
                r_sup=self.r_limits[i_layer + 1],
                name=layer_name,
            )
            layer_model.update_polynomials(polynomials=self.polynomials, i_layer=i_layer)
            layer_models.append(layer_model)

        solid_earth_parameters.model.structure_parameters.i_layer_icb = (
            solid_earth_parameters.model.structure_parameters.i_layer_icb
            if solid_earth_parameters.model.structure_parameters.i_layer_icb is not None
            else min(
                layer_index
                for layer_index, layer_name in enumerate(self.layer_names)
                if "FLUID" in layer_name
            )
        )
        solid_earth_parameters.model.structure_parameters.i_layer_cmb = (
            solid_earth_parameters.model.structure_parameters.i_layer_cmb
            if solid_earth_parameters.model.structure_parameters.i_layer_cmb is not None
            else max(
                layer_index
                for layer_index, layer_name in enumerate(self.layer_names)
                if "FLUID" in layer_name
            )
            + 1
        )

        return SolidEarthNumericalModel(
            name=name,
            layer_models=layer_models,
            solid_earth_parameters=solid_earth_parameters,
        )
