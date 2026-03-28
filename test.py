"""
All base functionalities. To test via pytest test.py.
"""

from pathlib import Path
from typing import Iterable, Optional

from base_models import (
    DEFAULT_MODELS,
    TEST_FIGURES_PATH,
    TEST_PATH,
    BoundaryCondition,
    Direction,
    SolidEarthModelPart,
    load_base_model,
    save_base_model,
)
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots, suptitle
from numpy import array, ndarray, zeros

from alna import (
    DEFAULT_COMPONENT_PARAMETERS,
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
    load_solid_earth_numerical_model,
)

TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH = TEST_PATH.joinpath(
    "solid_earth_model_profile_descriptions"
)
TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH = TEST_PATH.joinpath("solid_earth_numerical_models")
TEST_PARAMETER_FILE_PATH = Path(".")
TEST_SOLID_EARTH_PARAMETERS_PATH = TEST_PATH.joinpath("solid_earth_parameters")
ELASTIC_PERIOD_TAB = array(object=[1.0])  # (yr).
DEFAULT_REFERENCE_LOVE_NUMBERS_PATH = Path("../../ViscoLove/EARTH_MODELS/PREM_ELASTIC")
NUMERICAL_TOLERANCE = 5e-5


def test_load_solid_earth_model_profile_descriptions(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
) -> None:
    """
    Loads solid Earth model profile descriptions, then saves and reloads to verify consistency.
    """

    if models is None:

        models = DEFAULT_MODELS

    for solid_earth_model_part in SolidEarthModelPart:

        profile_description = SolidEarthModelDescription(
            name=models[solid_earth_model_part.value], solid_earth_model_part=solid_earth_model_part
        )
        profile_description.save(
            name=models[solid_earth_model_part.value] + "_save_test",
            path=test_path,
        )
        reloaded_profile_description = SolidEarthModelDescription(
            name=models[solid_earth_model_part.value] + "_save_test", path=test_path
        )

        assert profile_description.__dict__ == reloaded_profile_description.__dict__


def test_load_solid_earth_parameters(
    name: str = "parameters",
    path: Path = TEST_PARAMETER_FILE_PATH,
    test_path: Path = TEST_PATH,
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
        path=test_path,
    )
    reloaded_parameters: SolidEarthParameters = load_base_model(
        name=name, path=test_path, base_model_type=SolidEarthParameters
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

    assert model_1.love_numbers.keys() == model_2.love_numbers.keys()

    for love_numbers_1, love_numbers_2 in zip(
        model_1.love_numbers.values(), model_2.love_numbers.values()
    ):

        assert sum(abs(love_numbers_1 - love_numbers_2).flatten()) == 0.0

    assert model_1.love_number_partials.keys() == model_2.love_number_partials.keys()

    for partials_1, partials_2 in zip(
        model_1.love_number_partials.values(), model_2.love_number_partials.values()
    ):

        assert partials_1.keys() == partials_2.keys()

        for love_number_partials_1, love_number_partials_2 in zip(
            partials_1.values(), partials_2.values()
        ):

            assert sum(abs(love_number_partials_1 - love_number_partials_2).flatten()) == 0.0


def test_load_solid_earth_numerical_model(
    model: str = DEFAULT_MODELS[SolidEarthModelPart.ELASTIC.value],
    name: str = "parameters",
    path: Path = TEST_PARAMETER_FILE_PATH,
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
    path: Path = TEST_PARAMETER_FILE_PATH,
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

    for component in SolidEarthModelPart:

        if component == SolidEarthModelPart.ELASTIC:

            continue

        solid_earth_numerical_model.merge(
            solid_earth_model_description=SolidEarthModelDescription(
                name=models[component.value],
                solid_earth_model_part=component,
            ),
            name=models[component.value],
        )

    solid_earth_numerical_model.save(path=test_path)
    reloaded_solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=solid_earth_numerical_model.name, path=test_path
    )
    verify_solid_earth_numerical_model_consistency(
        model_1=solid_earth_numerical_model, model_2=reloaded_solid_earth_numerical_model
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


def compare_plot_to_elastic_reference(
    degrees_list: list[int],
    reference_love_numbers: ndarray,
    solid_earth_numerical_model: SolidEarthNumericalModel,
    path: Path = TEST_FIGURES_PATH,
) -> None:
    """
    Generates a figure of 3 subplots respectively for (h', l', k'), (h*, l*, k*) and (h, l, k).
    """

    axes: Iterable[Axes]
    figure, axes = subplots(3, figsize=(5, 7), sharex=True)
    figure_title = "Elastic Love numbers compared to reference"
    suptitle(figure_title)

    for ax, boundary_condition in zip(axes, BoundaryCondition):

        for label, direction in zip("hlk", Direction):

            has_zero_value = (
                boundary_condition != BoundaryCondition.LOAD
            ) or direction == Direction.POTENTIAL
            ax.loglog(
                degrees_list[(0 if not has_zero_value else 1) :],
                [
                    100
                    * abs(
                        (
                            love_numbers[0, boundary_condition.value, direction.value]
                            - reference_love_numbers[i_n, boundary_condition.value, direction.value]
                        )
                        / reference_love_numbers[i_n, boundary_condition.value, direction.value]
                    )
                    for i_n, love_numbers in enumerate(
                        solid_earth_numerical_model.love_numbers.values()
                    )
                    if not (i_n == 0 and has_zero_value)
                ],
                label=rf"$\Delta {label}"
                + (
                    "'"
                    if boundary_condition == BoundaryCondition.LOAD
                    else ("^*" if boundary_condition == BoundaryCondition.SHEAR else "")
                )
                + "_n$",
            )
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=5,
            width=1,
        )
        ax.set_ylabel("%")
        ax.legend(
            loc="upper center" if boundary_condition == BoundaryCondition.LOAD else "upper left",
            frameon=False,
        )

        if boundary_condition == BoundaryCondition.POTENTIAL:

            ax.set_xlabel("Degree")

    path.mkdir(exist_ok=True, parents=True)

    for file_format in ["svg", "png"]:

        figure.savefig(fname=path.joinpath(figure_title + "." + file_format), format=file_format)


def test_integrate_elastic(
    model: str = DEFAULT_MODELS[SolidEarthModelPart.ELASTIC.value],
    name: str = "parameters",
    path: Path = TEST_PARAMETER_FILE_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    reference_love_numbers_path: Path = DEFAULT_REFERENCE_LOVE_NUMBERS_PATH,
) -> None:
    """
    Loads an elastic solid Earth model profile description, generates the corresponding elastic
    solid Earth numerical model, integrates the y_i system, saves and reloads the love numbers to
    verify consistency, compares to reference Love numbers from litterature and saves the
    corresponding figure.
    """

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
        period_tab_per_degree={n: ELASTIC_PERIOD_TAB for n in degrees_list}
    )
    solid_earth_numerical_model.save(path=test_path)
    reloaded_solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=solid_earth_numerical_model.name, path=test_path
    )
    verify_solid_earth_numerical_model_consistency(
        model_1=solid_earth_numerical_model,
        model_2=reloaded_solid_earth_numerical_model,
    )
    compare_plot_to_elastic_reference(
        degrees_list=degrees_list,
        reference_love_numbers=reference_love_numbers,
        solid_earth_numerical_model=solid_earth_numerical_model,
    )

    for n, love_numbers in zip(degrees_list, reference_love_numbers):

        assert (
            sum(
                abs(
                    array(object=love_numbers - solid_earth_numerical_model.love_numbers[n][0])
                ).flatten()
            )
            / 9
            < NUMERICAL_TOLERANCE
        )


def test_check_anelastic_settings(
    models: Optional[dict[str, str]] = None,
    name: str = "parameters",
    path: Path = TEST_PARAMETER_FILE_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
) -> None:
    """
    Integrates a model in every different anelastic setting (2x2x2 = 8 options) to check robustness
    for a single (degree, period) pair.
    """

    if models is None:

        models = DEFAULT_MODELS

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

    # Merges with the other components.
    for component in SolidEarthModelPart:

        if component == SolidEarthModelPart.ELASTIC:

            continue

        solid_earth_numerical_model.merge(
            solid_earth_model_description=SolidEarthModelDescription(
                name=models[component.value],
                solid_earth_model_part=component,
            ),
            name=models[component.value],
        )

    # Loops on every option and tests the integration.
    for viscous_component in [False, True]:

        parameters.model.component_parameters.viscous_component = viscous_component

        for transient_component in [False, True]:

            parameters.model.component_parameters.transient_component = transient_component

            for bounded_attentuation_functions in [False, True]:

                parameters.model.component_parameters.bounded_attenuation_functions = (
                    bounded_attentuation_functions
                )
                solid_earth_numerical_model.solid_earth_parameters = parameters
                solid_earth_numerical_model.compute_love_numbers(
                    period_tab_per_degree={2: ELASTIC_PERIOD_TAB}
                )


# TODO:

# # Third function that plots time-dependent difference to elastic in 2D and save.

# Fourth function that does the 8 loop: the integration for every component parameters
# possibility and small degree list and period list. Test saving and loading.

# Add the
# Fifth function that also integrates partials with respect to transient or viscous parameters
# when possible. Test saving and loading. Compare with ultra-defined finite differences: 2 plots
# for k_2: 2D and 1D along alpha.
