"""
To produce Love numbers of interest and their partial deriavtives for a range of candidate physical
models.
"""

from pathlib import Path

from base_models import DATA_PATH, SolidEarthModelPart, load_base_model
from numpy import linspace, logspace

from alna import SolidEarthModelDescription, SolidEarthNumericalModel, SolidEarthParameters
from base_tests import TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH, initialize_test

INTEGRATION_PATH = DATA_PATH.joinpath("Love_numbers").joinpath("for_gins")
PERIODS_TAB = logspace(start=-2, stop=4, num=100, base=10)  # (yr).
DELTA_TAB = linspace(start=3, stop=15, num=13)
ALPHA_TAB = linspace(start=0.15, stop=0.3, num=16)
MODELS = {"elastic": "PREM", "attenuation": "Resovsky", "transient": "reference", "viscous": "VM7"}
PARAMETERS_NAME = "parameters"
PARAMETERS_PATH = Path(".")


if __name__ == "__main__":

    initialize_test(models=MODELS, test_path=INTEGRATION_PATH)
    profile_description = SolidEarthModelDescription(
        name=MODELS[SolidEarthModelPart.ELASTIC.value],
        solid_earth_model_part=SolidEarthModelPart.ELASTIC,
    )
    parameters: SolidEarthParameters = load_base_model(
        name=PARAMETERS_NAME, path=PARAMETERS_PATH, base_model_type=SolidEarthParameters
    )
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        profile_description.generate_solid_earth_numerical_model(
            name=MODELS[SolidEarthModelPart.ELASTIC.value], solid_earth_parameters=parameters
        )
    )
    solid_earth_numerical_model.merge_all(models=MODELS)
    solid_earth_numerical_model.compute_love_numbers(
        period_tab_per_degree={2: PERIODS_TAB, 3: PERIODS_TAB},
        parameters_to_invert_dictionary={
            r"\alpha^{MANTLE_0}": list(ALPHA_TAB),
            r"\Delta^{MANTLE_0}": list(DELTA_TAB),
        },
        path=TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    )
