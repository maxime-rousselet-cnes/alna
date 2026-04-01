"""
Tests the consistency of integration. To test via pytest integration_tests.py.
"""

from pathlib import Path
from shutil import rmtree
from typing import Optional

from base_models import DEFAULT_MODELS, SolidEarthModelPart, load_base_model
from numpy import array, linspace, logspace, ndarray, zeros

from alna import (
    DEFAULT_COMPONENT_PARAMETERS,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
    load_solid_earth_numerical_model,
)
from base_tests import (
    ALPHA_PERIOD_TAB,
    ELASTIC_PERIOD_TAB,
    TEST_PARAMETERS_SAVE_PATH,
    TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    verify_solid_earth_numerical_model_consistency,
)

N_PERIODS_VISCOUS_INTEGRATION_TEST = 20
VISCOUS_PERIOD_TAB = logspace(
    -3, 5, num=N_PERIODS_VISCOUS_INTEGRATION_TEST, base=10
)  # (yr), from sub-daily to 100 kyr.
DEFAULT_REFERENCE_LOVE_NUMBERS_PATH = Path("../../ViscoLove/EARTH_MODELS/PREM_ELASTIC")
NUMERICAL_TOLERANCE = 5e-5
TEST_ELASTIC_INTEGRATION_PATH = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath(
    "elastic_integration_test"
)
TEST_VISCOUS_INTEGRATION_PATH = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath(
    "viscous_integration_test"
)
TEST_ALPHA_PARTIAL_INTEGRATION_PATH = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath(
    "alpha_partial"
)
ALPHA_TAB = linspace(start=0.2, stop=0.3, num=100)


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
        period_tab_per_degree={n: ELASTIC_PERIOD_TAB for n in degrees_list}, format_name=False
    )
    solid_earth_numerical_model.save(path=test_path)
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


def test_integrate_viscous(
    models: Optional[dict[str, str]] = None,
    period_tab: ndarray = VISCOUS_PERIOD_TAB,
    elastic_test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    test_path: Path = TEST_VISCOUS_INTEGRATION_PATH,
) -> None:
    """
    Integrates a model in every different anelastic setting (2x2x2 = 8 options) to check robustness
    for a single (degree, period) pair.
    """

    if models is None:

        models = DEFAULT_MODELS

    if test_path.exists():

        rmtree(path=test_path)

    # Gets the elastic model.
    solid_earth_numerical_model: SolidEarthNumericalModel = load_solid_earth_numerical_model(
        name=models[SolidEarthModelPart.ELASTIC.value], path=elastic_test_path
    )
    solid_earth_numerical_model.merge_all(models=models)
    components = solid_earth_numerical_model.solid_earth_parameters.model.component_parameters
    components.viscous_component = True
    components.transient_component = False
    solid_earth_numerical_model.solid_earth_parameters.model.component_parameters = components
    solid_earth_numerical_model.compute_love_numbers(
        period_tab_per_degree={
            degree: period_tab for degree in solid_earth_numerical_model.love_numbers["real"].keys()
        }
    )
    solid_earth_numerical_model.save(path=test_path)


def test_integrate_alpha_partials(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_ALPHA_PARTIAL_INTEGRATION_PATH,
    periods_tab: ndarray = ALPHA_PERIOD_TAB,
    alpha_tab: ndarray = ALPHA_TAB,
) -> None:
    """
    Integrates the partial derivative of Love numbers with respect to alpha to compare it to finite.
    differences.
    """

    if models is None:

        models = DEFAULT_MODELS

    base_name = SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(models.values())

    if test_path.exists():

        rmtree(path=test_path)

    solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=base_name, path=test_path.parent, force_transient=True
    )

    for alpha in alpha_tab:

        solid_earth_numerical_model.expressions.terminal_parameter_values[r"\alpha^{MANTLE_0}"] = (
            alpha
        )
        solid_earth_numerical_model.name = SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(
            (base_name, f"_alpha_{alpha:.3f}")
        )
        solid_earth_numerical_model.compute_love_numbers(
            period_tab_per_degree={2: periods_tab}, parameters_to_invert=[r"\alpha^{MANTLE_0}"]
        )
        solid_earth_numerical_model.save(path=test_path)
