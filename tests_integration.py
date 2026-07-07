"""
Tests the consistency of integration. To test via pytest integration_tests.py.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from base_models import DEFAULT_MODELS, MODELS, SolidEarthModelPart, load_base_model
from numpy import array
from pytest import Config

from alna import (
    DEFAULT_COMPONENT_PARAMETERS,
    DEFAULT_REFERENCE_LOVE_NUMBERS_PATH,
    ELASTIC_PERIOD_TAB,
    NUMERICAL_TOLERANCE,
    TEST_ELASTIC_INTEGRATION_PATH,
    TEST_PARAMETERS_SAVE_PATH,
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
    compute_love_numbers_for_gins,
    initialize_test,
    load_reference_love_numbers_for_validation,
    load_solid_earth_numerical_model,
    partials_per_parameter_integration_tests,
    verify_solid_earth_numerical_model_consistency,
    viscous_model_integration_test,
)


def test_integrate_elastic(
    model: str = DEFAULT_MODELS[SolidEarthModelPart.ELASTIC.value],
    name: str = "parameters",
    path: Path = TEST_PARAMETERS_SAVE_PATH,
    test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    reference_love_numbers_path: Path = DEFAULT_REFERENCE_LOVE_NUMBERS_PATH,
) -> None:
    """
    Loads an elastic solid Earth model profile description, generates the corresponding elastic
    solid Earth numerical model, integrates the y_i system, saves and reloads the love numbers to
    verify consistency, compares to reference Love numbers from litterature and saves the
    corresponding figure.
    """

    initialize_test(models=None, test_path=test_path)
    degrees_list, reference_love_numbers = load_reference_love_numbers_for_validation(
        path=reference_love_numbers_path
    )
    profile_description = SolidEarthModelDescription(
        name=model, solid_earth_model_part=SolidEarthModelPart.ELASTIC
    )
    parameters: SolidEarthParameters = load_base_model(
        name=name, path=path, base_model_type=SolidEarthParameters
    )
    parameters.model.component_parameters = DEFAULT_COMPONENT_PARAMETERS
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        profile_description.generate_solid_earth_numerical_model(
            name=model, solid_earth_parameters=parameters
        )
    )
    solid_earth_numerical_model.compute_love_numbers(
        period_tab_per_degree={n: ELASTIC_PERIOD_TAB for n in degrees_list},
        format_name=False,
        path=test_path,
    )
    reloaded_solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=solid_earth_numerical_model.name, path=test_path
    )
    verify_solid_earth_numerical_model_consistency(
        model_1=solid_earth_numerical_model,
        model_2=reloaded_solid_earth_numerical_model,
    )

    for n, love_numbers in zip(degrees_list, reference_love_numbers):

        assert (
            sum(
                abs(
                    array(
                        object=love_numbers - solid_earth_numerical_model.love_numbers["real"][n][0]
                    )
                ).flatten()
            )
            / 9
            < NUMERICAL_TOLERANCE
        )


def test_integrate_viscous(test_config: Config) -> None:
    """
    Integrates a model to benchmark in the Maxwell setting.
    """

    viscous_model_integration_test(
        local_mode=test_config["local_mode"], n_periods=test_config["n_periods"]
    )


def test_integrate_partials_per_parameter(test_config: Config) -> None:
    """
    Integrates partiel to compare to finite differences, for an elastic parameter, a viscous
    parameter and a transient parameter.
    """

    partials_per_parameter_integration_tests(
        local_mode=test_config["local_mode"], n_partial_tests=test_config["n_parameter_values"]
    )


def test_compute_love_numbers_for_gins(
    test_config: Config | dict[str, int | bool],
    degrees: Optional[list[int]] = None,
    models: Optional[dict[str, str]] = None,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range of candidate
    physical models on alpha, Delta and tau_m parameters.
    """

    compute_love_numbers_for_gins(
        local_mode=test_config["local_mode"],
        n_parameter_values=test_config["n_parameter_values"],
        n_periods=test_config["n_periods"],
        degrees=degrees if degrees else [2],
        models=models,
    )


def parse_args() -> Namespace:
    """
    Parses the same 3 optional parameters for direct Python execution.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--local_mode",
        action="store_true",
        default=False,
        help="Run tests in local mode.",
    )
    parser.add_argument(
        "--n_parameter_values",
        type=int,
        default=2,
        help="Number of parameter values to test for GINS-ready Love numbers.",
    )
    parser.add_argument(
        "--n_periods",
        type=int,
        default=2,
        help="Number of periods to integrate the Love numbers at.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    test_compute_love_numbers_for_gins(
        test_config={
            "local_mode": args.local_mode,
            "n_parameter_values": args.n_parameter_values,
            "n_periods": args.n_periods,
        },
        degrees=[2],
        models=MODELS,
    )
