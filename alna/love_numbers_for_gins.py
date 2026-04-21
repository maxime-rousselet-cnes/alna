"""
To produce Love numbers of interest and their partial deriavtives for a range of candidate physical
models.
"""

from pathlib import Path
from typing import Optional

from base_models import (
    LOVE_NUMBERS_PATH,
    BoundaryCondition,
    Direction,
    SolidEarthModelPart,
    load_base_model,
    save_base_model,
)
from numpy import linspace, logspace, ndarray, zeros

from .load_solid_earth_model import load_solid_earth_numerical_model
from .parameters import (
    ComponentParameters,
    build_base_name,
    compose_name_with_invertible_parameters,
    format_name_function,
)
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


def load_love_numbers_for_gins(
    path: Path = LOVE_NUMBERS_PATH.joinpath("for_gins"),
    models: Optional[dict[str, str]] = None,
    periods_tab: ndarray = PERIODS_TAB,
    alpha_tab: ndarray = ALPHA_TAB,
    delta_tab: ndarray = DELTA_TAB,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Gets already computed Love numbers and their derivatives with respect to alpha and Delta.
    """

    if models is None:

        models = MODELS

    love_numbers = zeros(
        shape=(len(alpha_tab), len(delta_tab), 2, len(periods_tab), 2), dtype=complex
    )
    love_number_alpha_partials = zeros(
        shape=(len(alpha_tab), len(delta_tab), 2, len(periods_tab), 2), dtype=complex
    )
    love_number_delta_partials = zeros(
        shape=(len(alpha_tab), len(delta_tab), 2, len(periods_tab), 2), dtype=complex
    )

    for i_alpha, alpha in enumerate(alpha_tab):

        for i_delta, delta in enumerate(delta_tab):

            model = load_solid_earth_numerical_model(
                name=compose_name_with_invertible_parameters(
                    name=format_name_function(
                        name=build_base_name(models=models),
                        component_parameters=ComponentParameters(
                            viscous_component=True,
                            transient_component=True,
                            bounded_attenuation_functions=True,
                        ),
                    ),
                    parameters_to_invert=[r"\alpha^{MANTLE_0}", r"\Delta^{MANTLE_0}"],
                    invertible_parameter_tab=[alpha, delta],
                ),
                path=path,
            )

            for degree_index in [0, 1]:

                love_numbers[i_alpha, i_delta, degree_index] = (
                    model.love_numbers["real"][degree_index + 2][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        [Direction.VERTICAL.value, Direction.POTENTIAL.value],
                    ]
                    + 1j
                    * model.love_numbers["imag"][degree_index + 2][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        [Direction.VERTICAL.value, Direction.POTENTIAL.value],
                    ]
                )
                love_number_alpha_partials[i_alpha, i_delta, degree_index] = (
                    model.love_number_partials["real"][r"\alpha^{MANTLE_0}"][degree_index + 2][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        [Direction.VERTICAL.value, Direction.POTENTIAL.value],
                    ]
                    + 1j
                    * model.love_number_partials["imag"][r"\alpha^{MANTLE_0}"][degree_index + 2][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        [Direction.VERTICAL.value, Direction.POTENTIAL.value],
                    ]
                )

                love_number_alpha_partials[i_alpha, i_delta, degree_index] = (
                    model.love_number_partials["real"][r"\Delta^{MANTLE_0}"][degree_index + 2][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        [Direction.VERTICAL.value, Direction.POTENTIAL.value],
                    ]
                    + 1j
                    * model.love_number_partials["imag"][r"\Delta^{MANTLE_0}"][degree_index + 2][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        [Direction.VERTICAL.value, Direction.POTENTIAL.value],
                    ]
                )

    return love_numbers, love_number_alpha_partials, love_number_delta_partials
