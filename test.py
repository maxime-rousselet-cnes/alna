"""
All base functionalities. To test via pytest test.py.
"""

from pathlib import Path

from base_models import DEFAULT_MODELS, TEST_PATH, SolidEarthModelPart, load_base_model

from alna import SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH, SolidEarthModelDescription

TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH = TEST_PATH.joinpath(
    "solid_earth_model_profile_descriptions"
)


def test_load_solid_earth_parameters(
    models: dict[str, str] = DEFAULT_MODELS,
    test_path: Path = TEST_SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH,
) -> None:
    """
    Loads parameters, then saves and reloads to verify consistency.
    """

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

        assert profile_description.layer_names == reloaded_profile_description.layer_names
        assert profile_description.r_limits == reloaded_profile_description.r_limits
        assert (
            profile_description.optional_crust_values
            == reloaded_profile_description.optional_crust_values
        )
        assert profile_description.polynomials == reloaded_profile_description.polynomials


# TODO: Loads solid Earth model profile descriptions, then saves and reloads to verify consistency.
# TODO: Load an elastic solid Earth model profile description, generate the corresponding elastic
#       solid Earth numerical model, save, reload and verify consistency.
# TODO: Load an elastic solid Earth numerical model, integrate until CMB, update it with the other
#       components, save, reload and verify consistency.
# TODO: Load an elastic solid Earth numerical model, integrate until CMB, update it with the other
#       components, save, reload and verify consistency.
# TODO: Load an elastic solid Earth numerical model, integrate until CMB, update it with the other
#       components, save, reload and verify consistency.
