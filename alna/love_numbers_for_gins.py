"""
To produce Love numbers of interest and their partial deriavtives for a range of candidate physical
models.
"""

from pathlib import Path

from base_models import LOVE_NUMBERS_PATH, SolidEarthModelPart, load_base_model, save_base_model
from numpy import linspace, logspace, ndarray

from .solid_earth_model import (
    SolidEarthModelDescription,
    SolidEarthNumericalModel,
    SolidEarthParameters,
)

INTEGRATION_PATH = LOVE_NUMBERS_PATH.joinpath("for_gins")
PERIODS_TAB = logspace(start=-2, stop=4, num=40, base=10)  # (yr).
DELTA_TAB = linspace(start=3, stop=15, num=13)
ALPHA_TAB = linspace(start=0.15, stop=0.3, num=16)
MODELS = {"elastic": "PREM", "attenuation": "Resovsky", "transient": "reference", "viscous": "VM7"}
PARAMETERS_NAME = "parameters"
PARAMETERS_PATH = Path(".")


def compute_love_numbers_for_gins(
    alpha_tab: ndarray = ALPHA_TAB,
    delta_tab: ndarray = DELTA_TAB,
    periods_tab: ndarray = PERIODS_TAB,
    parameters_path: Path = PARAMETERS_PATH,
    parameters_file_name: str = PARAMETERS_NAME,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range of candidate
    physical models on alpha and delta parameters.
    """

    profile_description = SolidEarthModelDescription(
        name=MODELS[SolidEarthModelPart.ELASTIC.value],
        solid_earth_model_part=SolidEarthModelPart.ELASTIC,
    )
    parameters: SolidEarthParameters = load_base_model(
        name=parameters_file_name, path=parameters_path, base_model_type=SolidEarthParameters
    )
    solid_earth_numerical_model: SolidEarthNumericalModel = (
        profile_description.generate_solid_earth_numerical_model(
            name=MODELS[SolidEarthModelPart.ELASTIC.value], solid_earth_parameters=parameters
        )
    )
    solid_earth_numerical_model.merge_all(models=MODELS)
    solid_earth_numerical_model.compute_love_numbers(
        period_tab_per_degree={2: periods_tab, 3: periods_tab},
        parameters_to_invert_dictionary={
            r"\alpha^{MANTLE_0}": list(alpha_tab),
            r"\Delta^{MANTLE_0}": list(delta_tab),
        },
        path=INTEGRATION_PATH,
    )
    save_base_model(obj=periods_tab, name="periods_tab", path=INTEGRATION_PATH)
