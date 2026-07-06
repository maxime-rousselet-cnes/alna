"""
Defines all parameter classes.
"""

from dataclasses import dataclass
from itertools import product
from json import dumps
from pathlib import Path
from subprocess import run
from typing import Optional

from base_models import (
    DEFAULT_MODELS,
    DEFAULT_WORKDIR,
    EARTH_RADIUS,
    SolidEarthModelPart,
    save_base_model,
)
from numpy import linspace, logspace, ndarray
from pydantic import BaseModel

from .constants import (
    DEFAULT_PARAMETER_LINES_FILE_NAME,
    DEFAULT_PARAMETER_LINES_PATH,
    DEFAULT_PERIOD_TAB_PER_DEGREE_FILE_NAME,
    DEFAULT_PERIOD_TAB_PER_DEGREE_PATH,
    SOLID_EARTH_NUMERICAL_MODEL_NAME_FROM_INVERTIBLE_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
)


class ComponentParameters(BaseModel):
    """
    Options to choose rheology components to include in the model.
    Elasticity is not optional.
    """

    viscous_component: bool = False
    transient_component: bool = False
    # Unbounded attenuation functions can be used in the transient component to check consistency
    # with specific short-term Love numbers litterature. However, they produce non-physical values
    # at long-term.
    bounded_attenuation_functions: bool = False


DEFAULT_COMPONENT_PARAMETERS = ComponentParameters()


class StructureParameters(BaseModel):
    """
    Defines the solid Earth model parameters usefull for y_i system integration.
    """

    dynamic_term: bool = True  # Whether to use omega^2 terms in the y_i system or not.
    # Number of layers under boundaries. If they are None: Automatic detection using elasticity
    # model layer names.
    # Number of layers under the Inner-Core Boundary.
    i_layer_icb: Optional[int] = None  # Should be >= 0.
    # Number of total layers under the Mantle-Core Boundary.
    i_layer_cmb: Optional[int] = None  # Should be >= i_layer_icb.


DEFAULT_STRUCTURE_PARAMETERS = StructureParameters()


class SolidEarthModelParameters(BaseModel):
    """
    Parameterizes the solid Earth model.
    """

    component_parameters: ComponentParameters = DEFAULT_COMPONENT_PARAMETERS
    # Whether to use 'optional_crust_values' values specified in the component profile description
    # file or not.
    # Usefull to easily switch from ocenanic to continental crust parameters.
    optional_crust_values: Optional[bool] = None
    # Length unit (m). Will be set to the default constant if None.
    radius_unit: Optional[float] = None
    structure_parameters: StructureParameters = DEFAULT_STRUCTURE_PARAMETERS

    def __init_subclass__(cls, **kwargs):

        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        component_parameters: ComponentParameters = DEFAULT_COMPONENT_PARAMETERS,
        optional_crust_values: Optional[bool] = None,
        radius_unit: Optional[float] = None,
        structure_parameters: StructureParameters = (DEFAULT_STRUCTURE_PARAMETERS),
    ):

        super().__init__()

        self.component_parameters = (
            component_parameters
            if isinstance(component_parameters, ComponentParameters)
            else ComponentParameters(**component_parameters)
        )
        self.optional_crust_values = (
            False if optional_crust_values is None else optional_crust_values
        )
        self.radius_unit = EARTH_RADIUS if radius_unit is None else radius_unit
        self.structure_parameters = (
            structure_parameters
            if isinstance(structure_parameters, StructureParameters)
            else StructureParameters(**structure_parameters)
        )


DEFAULT_SOLID_EARTH_MODEL_PARAMETERS = SolidEarthModelParameters()


class IntegrationParameters(BaseModel):
    """
    Describes the parameters necessary for the numerical integration of the y_i system.
    """

    high_degrees_radius_sensibility: (
        float  # Integrates starting whenever x**n > high_degrees_radius_sensibility.
    ) = 1.0e-4
    minimal_radius: float = 1.0e3  # r ~= 0 km exact definition (m).
    minimal_layer_radius_factor: float = 1.0
    atol: float = 1.0e-14  # The solver keeps the local error estimates under atol + rtol * abs(yr).
    rtol: float = 1.0e-10  # See atol parameter description.


DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS = IntegrationParameters()


class SolidEarthParameters(BaseModel):
    """
    Defines all solid Earth algorithm parameters.
    """

    model: SolidEarthModelParameters = DEFAULT_SOLID_EARTH_MODEL_PARAMETERS
    n_max: Optional[int] = None
    integration_parameters: IntegrationParameters = DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS
    compute_partials: bool = True


DEFAULT_SOLID_EARTH_PARAMETERS = SolidEarthParameters()


# List ofthe possible non-elastic rheologies.
ALL_COMPONENT_PARAMETERS: list[ComponentParameters] = [
    ComponentParameters(
        viscous_component=True,
        transient_component=True,
        bounded_attenuation_functions=True,
    ),
    ComponentParameters(
        viscous_component=False,
        transient_component=True,
        bounded_attenuation_functions=True,
    ),
    ComponentParameters(
        viscous_component=True,
        transient_component=False,
        bounded_attenuation_functions=False,
    ),
]

# Elastic rheology.
ELASTIC_COMPONENT_PARAMETERS = ComponentParameters(
    viscous_component=False,
    transient_component=False,
    bounded_attenuation_functions=False,
)


def format_name_function(name: str, component_parameters: ComponentParameters) -> str:
    """
    Eventually renames the model for clarity and separability of tests in directories.
    """

    suffix = ""

    if not component_parameters.transient_component:

        suffix += SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR + "no_transient"

    elif not component_parameters.bounded_attenuation_functions:

        suffix += (
            SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR + "no_bounded_attenuation_functions"
        )

    if not component_parameters.viscous_component:

        suffix += SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR + "no_viscous"

    if not name.endswith(suffix):

        name += suffix

    return name


def compose_name_with_invertible_parameters(
    name: str, parameters_to_invert: list[str], invertible_parameters_tab: list[float]
) -> str:
    """
    Builds a model name characterized by a root and invertible parameter names and values.
    """

    for parameter, value in zip(parameters_to_invert, invertible_parameters_tab):

        name = (
            name
            + SOLID_EARTH_NUMERICAL_MODEL_NAME_FROM_INVERTIBLE_PARAMETERS_SEPARATOR
            + parameter
            + f"_{value:.2e}"
        )

    return name


def build_base_name(models: dict[str, str]) -> str:
    """
    A posteriori builds the name of an already merged model.
    """

    return SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(models.values())


def generate_parameter_lines(
    parameters: Optional[
        dict[str, list[str] | tuple[float, float, float] | tuple[float, float, float, float]]
    ] = None,
    parameter_lines_file_name: str = DEFAULT_PARAMETER_LINES_FILE_NAME,
    parameter_lines_path: Path = DEFAULT_PARAMETER_LINES_PATH,
) -> None:
    """
    Generates a (.JSONL) file to be read by the parallel computing script. Each line is a model run
    with a specific set of parameters.
    A list of values is kept unmodified for generation of tuplets.
    A triplet generates a linearly-spaced array.
    A quadruplet generates a log-spaced array.
    """

    if not parameters:

        parameters = {}

    parameter_lines_path.mkdir(exist_ok=True, parents=True)
    output = parameter_lines_path.joinpath(parameter_lines_file_name)

    all_parameter_values = []

    for _, parameter_values in parameters.items():

        if isinstance(parameter_values, (list, ndarray)):

            all_parameter_values.append(parameter_values)

        elif len(parameter_values) == 3:

            start, stop, num = parameter_values
            all_parameter_values.append(linspace(start=start, stop=stop, num=num))

        else:

            start, stop, num, base = parameter_values
            all_parameter_values.append(logspace(start=start, stop=stop, num=num, base=base))

    with output.open("w", encoding="utf-8") as f:

        for parameter_combination in product(*all_parameter_values):

            f.write(dumps(dict(zip(parameters.keys(), parameter_combination))) + "\n")


@dataclass
class LoveNumbersLauncher:
    """
    To Launch multi-parameter partial integrations in parallel.
    """

    name: str = DEFAULT_MODELS[SolidEarthModelPart.ELASTIC.value]
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH
    output_path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH
    parameter_lines_file_name: str = DEFAULT_PARAMETER_LINES_FILE_NAME
    parameter_lines_path: Path = DEFAULT_PARAMETER_LINES_PATH
    period_tab_per_degree_file_name: Path = DEFAULT_PERIOD_TAB_PER_DEGREE_FILE_NAME
    period_tab_per_degree_path: Path = DEFAULT_PERIOD_TAB_PER_DEGREE_PATH


def launch_love_numbers_computing(
    period_tab_per_degree: dict[int, ndarray],
    local_mode: bool = False,
    parameters: Optional[
        dict[str, list[str] | tuple[float, float, float] | tuple[float, float, float, float]]
    ] = None,
    love_numbers_launcher: LoveNumbersLauncher = LoveNumbersLauncher(),
    base_command: Optional[list[str]] = None,
) -> None:
    """
    Saves the needed informations to prepare single/parallel run(s) and runs Love numbers computing.
    base_command should at least include --name NAME for Solid Earth model.
    """

    save_base_model(
        obj=period_tab_per_degree,
        name=love_numbers_launcher.period_tab_per_degree_file_name,
        path=love_numbers_launcher.period_tab_per_degree_path,
    )
    generate_parameter_lines(
        parameters=parameters,
        parameter_lines_file_name=love_numbers_launcher.parameter_lines_file_name,
        parameter_lines_path=love_numbers_launcher.parameter_lines_path,
    )
    print()
    print(
        " ".join(
            [
                "python",
                "exe_love_numbers_jobs_launcher.py",
                "local" if local_mode else "submit",
                "--name",
                love_numbers_launcher.name,
                "--path",
                str(love_numbers_launcher.path),
                "--output_path",
                str(love_numbers_launcher.output_path),
                "--period_tab_per_degree",
                love_numbers_launcher.period_tab_per_degree_file_name,
                "--period_tab_per_degree_path",
                str(love_numbers_launcher.period_tab_per_degree_path),
                "--parameter_lines",
                love_numbers_launcher.parameter_lines_file_name,
                "--parameter_lines_path",
                str(love_numbers_launcher.parameter_lines_path),
            ]
        )
    )
    print()
    run(
        args=[
            "python",
            "exe_love_numbers_jobs_launcher.py",
            "local" if local_mode else "submit",
            "--name",
            love_numbers_launcher.name,
            "--path",
            str(love_numbers_launcher.path),
            "--output_path",
            str(love_numbers_launcher.output_path),
            "--period_tab_per_degree",
            love_numbers_launcher.period_tab_per_degree_file_name,
            "--period_tab_per_degree_path",
            str(love_numbers_launcher.period_tab_per_degree_path),
            "--parameter_lines",
            love_numbers_launcher.parameter_lines_file_name,
            "--parameter_lines_path",
            str(love_numbers_launcher.parameter_lines_path),
        ]
        + (base_command if base_command else []),
        cwd=DEFAULT_WORKDIR,
        check=True,
    )
