"""
All base functionalities. To test via pytest base_tests.py.
"""

from pathlib import Path
from typing import Optional

from base_models import DEFAULT_MODELS, SolidEarthModelPart, load_base_model, save_base_model
from numpy import ndarray

from alna import (
    ELASTIC_PERIOD_TAB,
    PARTIAL_PERIOD_TAB,
    TEST_PARAMETERS_FILE_PATH,
    TEST_PARAMETERS_SAVE_PATH,
    TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
    TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    LoveNumbersLauncher,
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
    build_base_name,
    initialize_test,
    launch_love_numbers_computing,
    load_solid_earth_numerical_model,
    verify_solid_earth_numerical_model_consistency,
)


def test_load_solid_earth_model_profile_descriptions(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
) -> None:
    """
    Loads solid Earth model profile descriptions, then saves and reloads to verify consistency.
    """

    models = initialize_test(models=models, test_path=test_path)

    for solid_earth_model_part in SolidEarthModelPart:

        path = test_path.joinpath(solid_earth_model_part.value)
        profile_description = SolidEarthModelDescription(
            name=models[solid_earth_model_part.value], solid_earth_model_part=solid_earth_model_part
        )
        profile_description.save(
            name=models[solid_earth_model_part.value] + "_save_test",
            path=path,
        )
        reloaded_profile_description = SolidEarthModelDescription(
            name=models[solid_earth_model_part.value] + "_save_test", path=path
        )

        assert profile_description.__dict__ == reloaded_profile_description.__dict__


def test_load_solid_earth_parameters(
    name: str = "parameters",
    path: Path = TEST_PARAMETERS_FILE_PATH,
    save_path: Path = TEST_PARAMETERS_SAVE_PATH,
) -> None:
    """
    Loads solid Earth model profile descriptions, then saves and reloads to verify consistency.
    """

    parameters: SolidEarthParameters = load_base_model(
        name=name, path=path, base_model_type=SolidEarthParameters
    )
    save_base_model(
        obj=parameters,
        name=name,
        path=save_path,
    )
    reloaded_parameters: SolidEarthParameters = load_base_model(
        name=name, path=save_path, base_model_type=SolidEarthParameters
    )

    assert parameters.__dict__ == reloaded_parameters.__dict__


def test_load_solid_earth_numerical_model(
    model: str = DEFAULT_MODELS[SolidEarthModelPart.ELASTIC.value],
    name: str = "parameters",
    path: Path = TEST_PARAMETERS_SAVE_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
) -> None:
    """
    Loads an elastic solid Earth model profile description, generates the corresponding elastic
    solid Earth numerical model, saves and reloads to verify consistency.
    """

    profile_description = SolidEarthModelDescription(
        name=model, solid_earth_model_part=SolidEarthModelPart.ELASTIC
    )
    parameters: SolidEarthParameters = load_base_model(
        name=name, path=path, base_model_type=SolidEarthParameters
    )
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        profile_description.generate_solid_earth_numerical_model(
            name=model, solid_earth_parameters=parameters
        )
    )
    solid_earth_numerical_model.save(path=test_path)
    reloaded_solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=solid_earth_numerical_model.name, path=test_path
    )
    verify_solid_earth_numerical_model_consistency(
        model_1=solid_earth_numerical_model, model_2=reloaded_solid_earth_numerical_model
    )


def test_merge_solid_earth_numerical_models(
    models: Optional[dict[str, str]] = None,
    name: str = "parameters",
    path: Path = TEST_PARAMETERS_SAVE_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
) -> None:
    """
    Loads an elastic solid Earth model profile description, generates the corresponding elastic
    solid Earth numerical model, merges with the default anelastic components, saves and reloads to
    verifiy consistency.
    """

    if models is None:

        models = DEFAULT_MODELS

    elastic_profile_description = SolidEarthModelDescription(
        name=models[SolidEarthModelPart.ELASTIC.value],
        solid_earth_model_part=SolidEarthModelPart.ELASTIC,
    )
    parameters: SolidEarthParameters = load_base_model(
        name=name, path=path, base_model_type=SolidEarthParameters
    )
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        elastic_profile_description.generate_solid_earth_numerical_model(
            name=models[SolidEarthModelPart.ELASTIC.value], solid_earth_parameters=parameters
        )
    )
    solid_earth_numerical_model.merge_all(models=models)
    solid_earth_numerical_model.save(path=test_path)
    reloaded_solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=solid_earth_numerical_model.name, path=test_path
    )
    verify_solid_earth_numerical_model_consistency(
        model_1=solid_earth_numerical_model, model_2=reloaded_solid_earth_numerical_model
    )


def test_check_anelastic_settings(
    models: Optional[dict[str, str]] = None,
    name: str = "parameters",
    path: Path = TEST_PARAMETERS_SAVE_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath("check_anelastic_setting"),
    periods_tab: ndarray = ELASTIC_PERIOD_TAB,
) -> None:
    """
    Integrates a model in every different anelastic setting to check robustness for a single
    (degree, period) pair.
    """

    models = initialize_test(models=models, test_path=test_path)

    # Initializes the elastic model.
    elastic_profile_description = SolidEarthModelDescription(
        name=models[SolidEarthModelPart.ELASTIC.value],
        solid_earth_model_part=SolidEarthModelPart.ELASTIC,
    )
    parameters: SolidEarthParameters = load_base_model(
        name=name, path=path, base_model_type=SolidEarthParameters
    )
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        elastic_profile_description.generate_solid_earth_numerical_model(
            name=models[SolidEarthModelPart.ELASTIC.value], solid_earth_parameters=parameters
        )
    )
    solid_earth_numerical_model.merge_all(models=models)
    initial_name = solid_earth_numerical_model.name

    # Loops on every option and tests the integration.
    for viscous_component in [False, True]:

        parameters.model.component_parameters.viscous_component = viscous_component

        for transient_component in [False, True]:

            parameters.model.component_parameters.transient_component = transient_component

            for bounded_attentuation_functions in [False, True] if transient_component else [False]:

                parameters.model.component_parameters.bounded_attenuation_functions = (
                    bounded_attentuation_functions
                )
                solid_earth_numerical_model.solid_earth_parameters = parameters
                solid_earth_numerical_model.name = initial_name
                solid_earth_numerical_model.save(path=test_path.parent)
                launch_love_numbers_computing(
                    period_tab_per_degree={2: periods_tab},
                    love_numbers_launcher=LoveNumbersLauncher(
                        name=initial_name, path=test_path.parent, output_path=test_path
                    ),
                    base_command=["--not_compute_partials"],
                )


def test_partials(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath("partials"),
    periods_tab: ndarray = PARTIAL_PERIOD_TAB,
) -> None:
    """
    Integrates the partial derivative of degree 2 Love numbers at given periods with respect to
    the alpha parameter describing the transient regime in the whole mantle.
    """

    models = initialize_test(models=models, test_path=test_path)
    launch_love_numbers_computing(
        period_tab_per_degree={2: periods_tab},
        parameters={r"\alpha^{MANTLE_0}": [0.2, 0.3], r"\eta_m^{UPPER-MANTLE_0}": [3e20, 3e21]},
        love_numbers_launcher=LoveNumbersLauncher(
            name=build_base_name(models=models),
            path=test_path.parent,
            output_path=test_path,
        ),
        base_command=["--compute_partials", "--force_transient", "--force_viscous"],
    )
