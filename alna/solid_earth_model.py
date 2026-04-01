"""
Solid Earth model description class for preprocessing.
"""

from pathlib import Path
from typing import Optional

from base_models import (
    SolidEarthModelPart,
    adaptive_runge_kutta_45,
    load_base_model,
    non_adaptive_runge_kutta_45,
    partial_symbols,
    save_base_model,
    vector_variation_equation,
)
from numpy import array, empty, inf, ndarray, pi, zeros
from pydantic import BaseModel, ConfigDict
from sympy import Expr, Matrix, Symbol, flatten, lambdify

from .constants import (
    COMPLEX_PARTS,
    INITIAL_Y_I,
    LAYERS_SEPARATOR,
    SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
    SYMPY_COMPILATION_MODULES_TRANSIENT_FRIENDLY,
    Y_I_STATE_FOR_SURFACE,
    Y_I_STATE_VECTOR_LINE,
    G,
    compute_omega_tab,
)
from .parameters import SolidEarthParameters
from .rheological_formulas import fluid_to_solid, solid_to_fluid
from .sub_models import Expressions, IntegrationContext, LayerModel


class SolidEarthNumericalModel(BaseModel):
    """
    Describes a solid Earth model numerically. Manages the pre-processing when instanciating and
    merging. Manages numerical integrations as methods.
    """

    name: str
    layer_models: list[LayerModel]
    solid_earth_parameters: SolidEarthParameters
    units: dict[str, float] = {}
    love_numbers: dict[str, dict[int, ndarray]] = {}
    love_number_partials: dict[str, dict[str, dict[int, ndarray]]] = {}
    expressions: Expressions = Expressions()
    # Unsaved attribute.

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
        i_layer_main: int = self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
        r_inf = self.layer_models[i_layer_main].r_inf
        i_layer_merging_component = 0
        new_layer_models = self.layer_models[:i_layer_main]
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

    def merge_all(self, models: dict[str, str]) -> None:
        """
        Merges the elastic component with the other components: attenuation, transient and viscous.
        """
        for component in SolidEarthModelPart:

            if component == SolidEarthModelPart.ELASTIC:

                continue

            self.merge(
                solid_earth_model_description=SolidEarthModelDescription(
                    name=models[component.value],
                    solid_earth_model_part=component,
                ),
                name=models[component.value],
            )

    def create_propagators(self) -> None:
        """
        Generates all needed symbols for the Y_i system symbolic definition. Takes care of the
        nondimensionalization.
        """

        i_layer_cmb: int = self.solid_earth_parameters.model.structure_parameters.i_layer_cmb
        self.units = {
            r"R": self.solid_earth_parameters.model.radius_unit,
            r"\rho_0": self.layer_models[i_layer_cmb].evaluate(
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
        self.units[r"\omega_{m-inf}"] = self.units[r"f"]
        self.units[r"\mu_{k1}"] = self.units[r"\mu_0"]
        self.units[r"\eta_k"] = self.units[r"\eta_m"]
        self.expressions.create_propagators(
            model=self.solid_earth_parameters.model,
            layer_models=self.layer_models,
            units=self.units,
        )

    def initialize_love_numbers_computing(
        self,
        period_tab_per_degree: dict[int, ndarray],
        parameters_to_invert: list[str],
    ) -> tuple[dict[str, list[Expr]], dict[str, Matrix]]:
        """
        Symbolic preprocessing providing compiled functions for numerical integration.
        """

        # Creates the y_i abstract system and updates all symbols and numerical values.
        self.create_propagators()
        partial_expressions_per_parameter, partials_matrix_per_parameter = {}, {}

        # Defines formally the partial derivative symbols.
        for parameter in parameters_to_invert:

            partial_expressions, partials_matrix_for_parameter = partial_symbols(
                parameter=self.expressions.parameter_expressions[parameter],
                state_vector_line=Y_I_STATE_VECTOR_LINE,
            )
            partial_expressions_per_parameter[parameter] = partial_expressions
            partials_matrix_per_parameter[parameter] = partials_matrix_for_parameter

        self.love_numbers = {
            part: {
                n: zeros(shape=(len(period_tab), 3, 3), dtype=float)
                for n, period_tab in period_tab_per_degree.items()
            }
            for part in COMPLEX_PARTS
        }
        self.love_number_partials = {
            part: {
                parameter: {
                    n: zeros(shape=(len(period_tab), 3, 3), dtype=float)
                    for n, period_tab in period_tab_per_degree.items()
                }
                for parameter in parameters_to_invert
            }
            for part in COMPLEX_PARTS
        }

        return partial_expressions_per_parameter, partials_matrix_per_parameter

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
                    self.expressions.expressions[r"x"],
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
                modules=SYMPY_COMPILATION_MODULES_TRANSIENT_FRIENDLY,
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
                    self.expressions.expressions[r"n"]: n,
                    self.expressions.expressions[r"\omega"]: omega / self.units[r"f"],
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
                            g_0_fluid_inf=self.expressions.evaluate(
                                expression=r"g_0^{layer_{" + str(i_layer) + "}}",
                                x=layer_model.r_inf / self.solid_earth_parameters.model.radius_unit,
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
                        g_0_fluid_sup=self.expressions.evaluate(
                            expression=r"g_0^{layer_{" + str(i_layer) + "}}",
                            x=layer_model.r_sup / self.solid_earth_parameters.model.radius_unit,
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

    def compute_love_numbers_from_surface_solution(
        self, integration_context: IntegrationContext
    ) -> None:
        """
        Applies the Love number expressions to the integrated y_i at surface.
        """

        self.expressions.expressions[r"L_n^{\omega}"] = self.expressions.expressions[r"L_n"]

        for y_i_symbols, y_i_tab in zip(Y_I_STATE_FOR_SURFACE, integration_context.y_tabs[-1]):

            # Applies the expression of the solution to the (3, 6) y_i.
            self.expressions.expressions[r"L_n^{\omega}"] = self.expressions.expressions[
                r"L_n^{\omega}"
            ].xreplace(rule=dict(zip(y_i_symbols, y_i_tab[-1])))

        love_numbers = array(
            object=self.expressions.expressions[r"L_n^{\omega}"].doit(), dtype=complex
        )
        self.love_numbers["real"][integration_context.n][integration_context.i_omega, :, :] = array(
            object=love_numbers.real, dtype=float
        )

        if array(object=abs(love_numbers.imag)).any():

            self.love_numbers["imag"][integration_context.n][integration_context.i_omega, :, :] = (
                array(object=love_numbers.imag, dtype=float)
            )

    def compute_love_number_partials(
        self,
        integration_context: IntegrationContext,
        parameter: str,
        y_i_all_partial_symbols: dict[str, list[list[Expr]]],
        y_i_partials: ndarray,
    ) -> None:
        """
        Deduces the Love number partial derivates from the y_i partial derivative surface solutions.
        """

        love_number_partials = self.expressions.expressions[
            r"\frac{\partial L_n}{\partial " + parameter + "}"
        ]

        for y_i_symbols, y_tab_last_layer, y_i_partial_symbols, y_i_partial in zip(
            Y_I_STATE_FOR_SURFACE,
            integration_context.y_tabs[-1],
            y_i_all_partial_symbols[parameter],
            y_i_partials,
        ):

            love_number_partials = love_number_partials.xreplace(
                rule=dict(zip(y_i_symbols, y_tab_last_layer[-1]))
                | dict(zip(y_i_partial_symbols, y_i_partial))
            )

        love_number_partials_array = array(
            object=love_number_partials,
            dtype=complex,
        )
        self.love_number_partials["real"][parameter][integration_context.n][
            integration_context.i_omega, :, :
        ] = array(object=love_number_partials_array.real, dtype=float)

        if array(object=love_number_partials_array.imag).any():

            self.love_number_partials["imag"][parameter][integration_context.n][
                integration_context.i_omega, :, :
            ] = array(object=love_number_partials_array.imag, dtype=float)

    def integrate_partials(
        self,
        integration_context: IntegrationContext,
        parameter: str,
        partial_expressions_per_parameter: dict[str, list[Expr]],
        y_i_all_partial_symbols: dict[str, list[list[Expr]]],
    ) -> None:
        """
        Performs the quadrature of partial derivatives of the y_i system with respect to invertible
        parameters and deduce the Love number partial derivatives, for a single parameter, degree
        and period.
        """

        y_i_partials = zeros(shape=(3, 6), dtype=complex)

        # So layer_model actually represents the (i_layer + i_layer_cmb)-th layer.
        for i_layer, layer_model in enumerate(
            self.layer_models[self.solid_earth_parameters.model.structure_parameters.i_layer_cmb :]
        ):

            partial_propagator = layer_model.partial_propagators[parameter].xreplace(
                rule={
                    self.expressions.expressions[r"n"]: integration_context.n,
                    # Uses unitless pulsation for integration.
                    self.expressions.expressions[r"\omega"]: integration_context.omega
                    / self.units[r"f"],
                }
            )

            for i_boundary_condition, (x_tab, y_tab) in enumerate(
                zip(integration_context.x_tabs[i_layer], integration_context.y_tabs[i_layer])
            ):

                y_i_partials[i_boundary_condition] = non_adaptive_runge_kutta_45(
                    fun=lambdify(
                        args=[
                            self.expressions.expressions[r"x"],
                            Y_I_STATE_VECTOR_LINE,
                            partial_expressions_per_parameter[parameter],
                        ],
                        expr=flatten(partial_propagator),
                        modules=SYMPY_COMPILATION_MODULES_TRANSIENT_FRIENDLY,
                    ),
                    t=x_tab,
                    y_0=y_i_partials[i_boundary_condition],
                    parameters=y_tab,
                )[
                    -1
                ]  # The x-dependent partial is useless.

        self.compute_love_number_partials(
            integration_context=integration_context,
            parameter=parameter,
            y_i_all_partial_symbols=y_i_all_partial_symbols,
            y_i_partials=y_i_partials,
        )

    def compute_love_numbers_for_degree(
        self,
        n: int,
        period_tab: ndarray,
        parameters_to_invert: list[str],
        partial_expressions_per_parameter: dict[str, list[Expr]],
    ) -> None:
        """
        Performs the y_i system integration for a single degree n.
        """

        # Generates the Love number expression from y_i at x = 1. Depends numerically on n.
        self.expressions.define_love_number_expressions(n=n)
        y_i_all_partial_symbols: dict[str, list[list[Expr]]] = {}

        for parameter in parameters_to_invert:

            y_i_all_partial_symbols[parameter] = []
            self.expressions.expressions[r"\frac{\partial L_n}{\partial " + parameter + "}"] = (
                -2  # Because same term will be added three times below.
                * (
                    self.expressions.expressions[r"L_n"].diff(
                        self.expressions.parameter_expressions[parameter]
                    )
                )
            )

            for y_i_state_line_for_surface in Y_I_STATE_FOR_SURFACE:

                partial_expressions, partials_matrix_for_parameter = partial_symbols(
                    parameter=self.expressions.parameter_expressions[parameter],
                    state_vector_line=y_i_state_line_for_surface,
                )
                self.expressions.expressions[
                    r"\frac{\partial L_n}{\partial " + parameter + "}"
                ] += vector_variation_equation(
                    dynamic=self.expressions.expressions[r"L_n"],
                    parameter=self.expressions.parameter_expressions[parameter],
                    partials=partials_matrix_for_parameter,
                    state_vector_line=y_i_state_line_for_surface,
                ).reshape(
                    rows=3, cols=3
                )
                y_i_all_partial_symbols[parameter] += [partial_expressions]

            self.expressions.expressions[r"\frac{\partial L_n}{\partial " + parameter + "}"] = (
                self.expressions.evaluate(
                    expression=self.expressions.expressions[
                        r"\frac{\partial L_n}{\partial " + parameter + "}"
                    ],
                    x=1,
                )
            ).doit()

        # Apply numerical values so that only the (3, 6) y_i remain.
        self.expressions.expressions[r"L_n"] = self.expressions.evaluate(
            expression=self.expressions.expressions[r"L_n"],
            x=1,
        )

        for i_omega, omega in enumerate(compute_omega_tab(period_tab=period_tab)):

            # Integrates the raw y_i system (no partials) through every layer for the 3 boundary
            # conditions.
            x_tabs, y_tabs = self.integrate_y_i_system(n=n, omega=omega)
            integration_context = IntegrationContext(
                n=n, i_omega=i_omega, omega=omega, x_tabs=x_tabs, y_tabs=y_tabs
            )
            self.compute_love_numbers_from_surface_solution(integration_context=integration_context)

            for parameter in parameters_to_invert:

                self.integrate_partials(
                    integration_context=integration_context,
                    parameter=parameter,
                    partial_expressions_per_parameter=partial_expressions_per_parameter,
                    y_i_all_partial_symbols=y_i_all_partial_symbols,
                )

    def format_name(self) -> None:
        """
        Eventually renames the model for clarity and separability of tests in directories.
        """

        suffix = ""

        if not self.solid_earth_parameters.model.component_parameters.transient_component:

            suffix += SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR + "no_transient"

        elif (
            not self.solid_earth_parameters.model.component_parameters.bounded_attenuation_functions
        ):

            suffix += (
                SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR
                + "no_bounded_attenuation_functions"
            )

        if not self.solid_earth_parameters.model.component_parameters.viscous_component:

            suffix += SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR + "no_viscous"

        if not self.name.endswith(suffix):

            self.name += suffix

    def compute_love_numbers(
        self,
        period_tab_per_degree: dict[int, ndarray],  # (yr).
        parameters_to_invert: Optional[list[str]] = None,
        format_name: bool = True,
    ) -> None:
        """
        Performs the y_i system integration for every degree and period of the list.
        """

        if format_name:

            self.format_name()

        if parameters_to_invert is None:

            parameters_to_invert = []

        partial_expressions_per_parameter, partials_matrix_per_parameter = (
            self.initialize_love_numbers_computing(
                period_tab_per_degree=period_tab_per_degree,
                parameters_to_invert=parameters_to_invert,
            )
        )

        # Applies the variation equations on the y_i system for every invertible parameter and
        # replaces all variables by their numerical values in the expressions. Only remain x, omega,
        # n, the 6 y_i and their partial derivatives.
        for layer_model in self.layer_models:

            layer_model.partial_propagators = {}

            for parameter in parameters_to_invert:

                layer_model.partial_propagators[parameter] = self.expressions.evaluate(
                    expression=vector_variation_equation(
                        dynamic=layer_model.propagator,
                        parameter=self.expressions.parameter_expressions[parameter],
                        partials=partials_matrix_per_parameter[parameter],
                        state_vector_line=Y_I_STATE_VECTOR_LINE,
                    ),
                    component_parameters=self.solid_earth_parameters.model.component_parameters,
                )

            layer_model.propagator = self.expressions.evaluate(
                expression=layer_model.propagator,
                component_parameters=self.solid_earth_parameters.model.component_parameters,
            )

        for n, period_tab in period_tab_per_degree.items():

            self.compute_love_numbers_for_degree(
                n=n,
                period_tab=period_tab,
                parameters_to_invert=parameters_to_invert,
                partial_expressions_per_parameter=partial_expressions_per_parameter,
            )


def load_solid_earth_numerical_model(
    name: str,
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    force_transient: Optional[bool] = None,
) -> SolidEarthNumericalModel:
    """
    Loads a solid Earth numerical model and formats its expressions.
    """

    loaded_content = load_base_model(name=name, path=path)
    love_numbers: dict[str, dict[int, list[list[list[float]]]]] = loaded_content["love_numbers"]
    love_number_partials: dict[str, dict[str, dict[int, list[list[list[float]]]]]] = loaded_content[
        "love_number_partials"
    ]
    solid_earth_numerical_model = SolidEarthNumericalModel(
        name=loaded_content["name"],
        layer_models=[LayerModel() for _ in range(len(loaded_content["layer_models"]))],
        solid_earth_parameters=loaded_content["solid_earth_parameters"],
        units=loaded_content["units"],
        love_numbers={
            part: {
                n: array(object=love_numbers_tab)
                for n, love_numbers_tab in love_numbers_part.items()
            }
            for part, love_numbers_part in love_numbers.items()
        },
        love_number_partials={
            part: {
                parameter: {
                    n: array(object=love_number_partials_tab)
                    for n, love_number_partials_tab in partials.items()
                }
                for parameter, partials in love_number_partials_part.items()
            }
            for part, love_number_partials_part in love_number_partials.items()
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

    if force_transient is not None:

        model = solid_earth_numerical_model.solid_earth_parameters.model
        model.component_parameters.transient_component = force_transient
        solid_earth_numerical_model.solid_earth_parameters.model = model

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
    #       - \omega_{m-inf}: (Hz).
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
