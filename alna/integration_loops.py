"""
Tests the consistency of integration. To test via pytest integration_tests.py.
"""

from pathlib import Path
from shutil import rmtree
from typing import Optional

from base_models import DEFAULT_MODELS, SolidEarthModelPart, load_base_model, save_base_model
from numpy import array, logspace, ndarray, zeros
from pydantic import BaseModel, ConfigDict

from .constants import (
    COMPLEX_PARTS,
    DEFAULT_PARAMETERS_NAME,
    ELASTIC_PERIOD_TAB,
    PARTIAL_PERIOD_TAB,
    ROOT_PATH,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
    TEST_ELASTIC_INTEGRATION_PATH,
    TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
)
from .load_solid_earth_model import load_solid_earth_numerical_model
from .parameters import LoveNumbersLauncher, launch_love_numbers_computing
from .solid_earth_model import (
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
)

NUMERICAL_TOLERANCE = 5e-5
DEFAULT_LOG10_PERIODS_FOR_VISCOUS_INTEGRATION_TEST_LOWER_BOUND = -3
DEFAULT_LOG10_PERIODS_FOR_VISCOUS_INTEGRATION_TEST_UPPER_BOUND = 5
DEFAULT_BASE_COMMAND = ["--compute_partials", "--force_viscous", "--force_transient"]
ALPHA_LOWER_BOUND = 0.1
ALPHA_UPPER_BOUND = 0.4
LOG10_DELTA_LOWER_BOUND = -2.0
LOG10_DELTA_UPPER_BOUND = 1.0
DEFAULT_FOR_GINS_OUTPUT_DIRECTORY = "for_gins"


def initialize_test(models: Optional[dict[str, str]], test_path: Path) -> dict[str, str]:
    """
    Heavy tests always starts by this initialization.
    """

    if models is None:

        models = DEFAULT_MODELS

    if test_path.exists():

        rmtree(path=test_path)

    return models


def verify_solid_earth_numerical_model_consistency(
    model_1: SolidEarthNumericalModel, model_2: SolidEarthNumericalModel
) -> None:
    """
    Verifies the consistency in attributes down to symbols.
    """

    assert model_1.name == model_2.name
    assert model_1.solid_earth_parameters == model_2.solid_earth_parameters
    assert model_1.units == model_2.units
    # Does not verify consistency on expressions.
    assert len(model_1.layer_models) == len(model_2.layer_models)

    for layer_model, reloaded_layer_model in zip(model_1.layer_models, model_2.layer_models):

        layer_model.propagator = None
        reloaded_layer_model.propagator = None
        layer_model.partial_propagators = None
        reloaded_layer_model.partial_propagators = None
        assert layer_model.__dict__ == reloaded_layer_model.__dict__

    if model_1.love_numbers or model_2.love_numbers:

        for part in COMPLEX_PARTS:

            assert model_1.love_numbers[part].keys() == model_2.love_numbers[part].keys()

            for love_numbers_1, love_numbers_2 in zip(
                model_1.love_numbers[part].values(), model_2.love_numbers[part].values()
            ):

                assert sum(array(object=abs(love_numbers_1 - love_numbers_2)).flatten()) == 0.0

            assert (
                model_1.love_number_partials[part].keys()
                == model_2.love_number_partials[part].keys()
            )

            for partials_1, partials_2 in zip(
                model_1.love_number_partials[part].values(),
                model_2.love_number_partials[part].values(),
            ):

                assert partials_1.keys() == partials_2.keys()

                for love_number_partials_1, love_number_partials_2 in zip(
                    partials_1.values(), partials_2.values()
                ):

                    assert (
                        sum(
                            array(
                                object=abs(love_number_partials_1 - love_number_partials_2)
                            ).flatten()
                        )
                        == 0.0
                    )


def load_reference_love_number_file_for_validation(file_path: Path) -> tuple[list[int], ndarray]:
    """
    Load a single Love number file (H, L, or K).
    """

    degrees = []
    values = []

    with file_path.open("r") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):

                continue

            parts = line.split()

            if len(parts) < 2:

                continue

            degrees += [int(parts[0])]
            val_str = parts[1].lower()
            values += [0.0 if val_str in ("+nan", "nan") else float(val_str)]

    return degrees, array(object=values, dtype=float)


def load_reference_love_numbers_for_validation(path: Path) -> tuple[list[int], ndarray]:
    """
    Load all A. Michel Love numbers into a (n, 3, 3) array.
    """

    types = ["L", "S", "P"]
    components = ["H", "L", "K"]
    file_map = {
        (i, j): path / f"{t}LN_{c}.txt"
        for i, t in enumerate(types)
        for j, c in enumerate(components)
    }
    first_key = next(iter(file_map))
    degrees, _ = load_reference_love_number_file_for_validation(file_map[first_key])
    n = len(degrees)
    love_numbers = zeros((n, 3, 3), dtype=float)

    for (i, j), file_path in file_map.items():

        _, arr = load_reference_love_number_file_for_validation(file_path)

        if len(arr) != n:

            raise ValueError(f"Inconsistent number of degrees in {file_path.name}")

        love_numbers[:, i, j] = arr

    return degrees, love_numbers


def viscous_model_integration_test(
    local_mode: bool = True,
    n_periods: int = 2,
    models: Optional[dict[str, str]] = None,
    elastic_test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
) -> None:
    """
    Integrates a model to benchmark in the Maxwell setting.
    """

    periods_tab: ndarray = logspace(
        start=DEFAULT_LOG10_PERIODS_FOR_VISCOUS_INTEGRATION_TEST_LOWER_BOUND,
        stop=DEFAULT_LOG10_PERIODS_FOR_VISCOUS_INTEGRATION_TEST_UPPER_BOUND,
        num=n_periods,
        base=10,
    )
    viscous_integration_test_path = test_path.joinpath("viscous")
    models = initialize_test(models=models, test_path=viscous_integration_test_path)
    solid_earth_numerical_model: SolidEarthNumericalModel = load_solid_earth_numerical_model(
        name=models[SolidEarthModelPart.ELASTIC.value], path=elastic_test_path
    )
    solid_earth_numerical_model.merge_all(models=models)
    solid_earth_numerical_model.save(path=viscous_integration_test_path)
    save_base_model(obj=periods_tab, name="periods_tab", path=viscous_integration_test_path)
    launch_love_numbers_computing(
        period_tab_per_degree={
            degree: periods_tab
            for degree in solid_earth_numerical_model.love_numbers["real"].keys()
        },
        local_mode=local_mode,
        love_numbers_launcher=LoveNumbersLauncher(
            name=solid_earth_numerical_model.name,
            path=viscous_integration_test_path,
            output_path=viscous_integration_test_path,
        ),
        base_command=["--not_compute_partials", "--force_viscous"],
    )


class MultiParametersLoop(BaseModel):
    """
    To manage defautl parameters for multi-parameters Love numbers integration loop.
    """

    degrees: list[int] | ndarray = [2]
    periods: list[int] | ndarray = ELASTIC_PERIOD_TAB
    parameters: Optional[
        dict[str, list[float] | tuple[float, float, int] | tuple[float, float, int, float]]
    ] = None
    parameters_path: Path = ROOT_PATH
    parameters_file_name: str = DEFAULT_PARAMETERS_NAME
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH
    output_directory: str = DEFAULT_FOR_GINS_OUTPUT_DIRECTORY

    model_config = ConfigDict(arbitrary_types_allowed=True)  # To authorize arrays.

    def get_single_parameter(self) -> str:
        """
        For single parameter integration tests.
        """

        assert len(self.parameters) == 1
        parameter = list(self.parameters.keys())[0]
        self.output_directory = parameter

        return parameter

    def set_periods(self, periods: list[float] | ndarray) -> None:
        """
        For single parameter integration tests.
        """

        self.periods = periods


def multi_parameter_integration(
    local_mode: bool = True,
    multi_parameter_love_numbers_loop: MultiParametersLoop = MultiParametersLoop(),
    base_command: Optional[list[str]] = None,
    models: Optional[dict[str, str]] = None,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range of candidate
    physical models on alpha and Delta parameters.
    """

    if not models:

        models = DEFAULT_MODELS

    profile_description = SolidEarthModelDescription(
        name=models[SolidEarthModelPart.ELASTIC.value],
        solid_earth_model_part=SolidEarthModelPart.ELASTIC,
    )
    parameters: SolidEarthParameters = load_base_model(
        name=multi_parameter_love_numbers_loop.parameters_file_name,
        path=multi_parameter_love_numbers_loop.parameters_path,
        base_model_type=SolidEarthParameters,
    )
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        profile_description.generate_solid_earth_numerical_model(
            name=models[SolidEarthModelPart.ELASTIC.value], solid_earth_parameters=parameters
        )
    )
    save_base_model(
        obj=multi_parameter_love_numbers_loop.periods,
        name="periods_tab",
        path=multi_parameter_love_numbers_loop.path.joinpath(
            multi_parameter_love_numbers_loop.output_directory
        ),
    )
    solid_earth_numerical_model.merge_all(models=models)
    solid_earth_numerical_model.save(path=multi_parameter_love_numbers_loop.path)
    launch_love_numbers_computing(
        period_tab_per_degree={
            degree: multi_parameter_love_numbers_loop.periods
            for degree in multi_parameter_love_numbers_loop.degrees
        },
        local_mode=local_mode,
        parameters=multi_parameter_love_numbers_loop.parameters,
        love_numbers_launcher=LoveNumbersLauncher(
            name=solid_earth_numerical_model.name,
            path=multi_parameter_love_numbers_loop.path,
            output_path=multi_parameter_love_numbers_loop.path.joinpath(
                multi_parameter_love_numbers_loop.output_directory
            ),
        ),
        base_command=base_command if base_command is not None else DEFAULT_BASE_COMMAND,
    )


def partial_integration_test_per_parameter(
    local_mode: bool = True,
    base_command: Optional[list[str]] = None,
    periods: ndarray = PARTIAL_PERIOD_TAB,
    models: Optional[dict[str, str]] = None,
    multi_parameter_love_numbers_loop: MultiParametersLoop = MultiParametersLoop(),
) -> None:
    """
    Integrates the partial derivative of Love numbers with respect to a parameter to compare it
    later to finite differences.
    """

    parameter = multi_parameter_love_numbers_loop.get_single_parameter()
    models = initialize_test(
        models=models, test_path=multi_parameter_love_numbers_loop.path.joinpath(parameter)
    )
    multi_parameter_love_numbers_loop.set_periods(periods=periods)
    multi_parameter_integration(
        local_mode=local_mode,
        multi_parameter_love_numbers_loop=multi_parameter_love_numbers_loop,
        base_command=base_command,
        models=models,
    )


def partials_per_parameter_integration_tests(
    local_mode: bool = True,
    n_partial_tests: int = 2,
    path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    parameters_path: Path = ROOT_PATH,
    parameters_file_name: str = DEFAULT_PARAMETERS_NAME,
) -> None:
    """
    Integrates partials to compare to finite differences, for an elastic parameter, a viscous
    parameter and transient parameters. Depends only on whether the mode is local or not, and the
    wanted length for integration tabs.
    """

    partial_integration_test_per_parameter(
        local_mode=local_mode,
        base_command=["--compute_partials", "--force_not_transient", "--force_not_viscous"],
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            periods=ELASTIC_PERIOD_TAB,
            parameters={r"\rho_0^{LOWER-MANTLE-1_0}": (7000.0, 9000.0, n_partial_tests)},
            parameters_path=parameters_path,
            parameters_file_name=parameters_file_name,
            path=path,
        ),
    )
    partial_integration_test_per_parameter(
        local_mode=local_mode,
        base_command=["--compute_partials", "--force_not_transient", "--force_viscous"],
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            periods=PARTIAL_PERIOD_TAB,
            parameters={r"\eta_m^{UPPER-MANTLE_0}": (18.0, 19.0, n_partial_tests, 10)},
            parameters_path=parameters_path,
            parameters_file_name=parameters_file_name,
            path=path,
        ),
    )
    partial_integration_test_per_parameter(
        local_mode=local_mode,
        base_command=["--compute_partials", "--force_transient", "--force_not_viscous"],
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            periods=PARTIAL_PERIOD_TAB,
            parameters={
                r"\alpha^{MANTLE_0}": (ALPHA_LOWER_BOUND, ALPHA_UPPER_BOUND, n_partial_tests)
            },
            parameters_path=parameters_path,
            parameters_file_name=parameters_file_name,
            path=path,
        ),
    )
    partial_integration_test_per_parameter(
        local_mode=local_mode,
        base_command=["--compute_partials", "--force_transient", "--force_not_viscous"],
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            periods=PARTIAL_PERIOD_TAB,
            parameters={
                r"\Delta^{MANTLE_0}": (
                    LOG10_DELTA_LOWER_BOUND,
                    LOG10_DELTA_UPPER_BOUND,
                    n_partial_tests,
                    10,
                )
            },
            parameters_path=parameters_path,
            parameters_file_name=parameters_file_name,
            path=path,
        ),
    )
