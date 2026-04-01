"""
All base functionalities. To test via pytest base_tests.py.
"""

from pathlib import Path
from shutil import rmtree
from typing import Optional

from base_models import (
    DEFAULT_MODELS,
    TEST_PATH,
    SolidEarthModelPart,
    load_base_model,
    save_base_model,
)
from numpy import array, ndarray

from alna import (
    COMPLEX_PARTS,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
    load_solid_earth_numerical_model,
)

TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH = TEST_PATH.joinpath(
    "solid_earth_model_profile_descriptions"
)
TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH = TEST_PATH.joinpath("solid_earth_numerical_models")
TEST_PARAMETERS_FILE_PATH = Path(".")
TEST_PARAMETERS_SAVE_PATH = TEST_PATH.joinpath("solid_earth_parameters")
ELASTIC_PERIOD_TAB = array(object=[1.0])  # (yr).
ALPHA_PERIOD_TAB = array(object=[10.0])  # (yr).


def test_load_solid_earth_model_profile_descriptions(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
) -> None:
    """
    Loads solid Earth model profile descriptions, then saves and reloads to verify consistency.
    """

    if models is None:

        models = DEFAULT_MODELS

    if test_path.exists():

        rmtree(path=test_path)

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


def _test_check_anelastic_settings(
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

    if models is None:

        models = DEFAULT_MODELS

    if test_path.exists():

        rmtree(path=test_path)

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
                solid_earth_numerical_model.compute_love_numbers(
                    period_tab_per_degree={2: periods_tab}
                )
                solid_earth_numerical_model.save(path=test_path)


def test_alpha_partial(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath("alpha_partial"),
    periods_tab: ndarray = ALPHA_PERIOD_TAB,
) -> None:
    """
    Integrates the partial derivative of degree 2 Love numbers at given periods with respect to
    the alpha parameter describing the transient regime in the whole mantle.
    """

    if models is None:

        models = DEFAULT_MODELS

    if test_path.exists():

        rmtree(path=test_path)

    solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(models.values()),
        path=test_path.parent,
        force_transient=True,
    )
    solid_earth_numerical_model.compute_love_numbers(
        period_tab_per_degree={2: periods_tab}, parameters_to_invert=[r"\alpha^{MANTLE_0}"]
    )
    solid_earth_numerical_model.save(path=test_path)


# TODO: Profile the alpha partial integration.
