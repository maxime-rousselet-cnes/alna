"""
To produce Love numbers of interest and their partial deriavtives for a candidate range of degrees,
periods and physical models.
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
from numpy import array, linspace, log, logspace, ndarray, zeros

from .constants import TEST_ELASTIC_INTEGRATION_PATH
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
LOVE_NUMBERS_FOR_GINS_TABS = {
    "degrees": [2],
    "periods": logspace(start=-2, stop=4, num=40, base=10),  # (yr).
    "alpha": linspace(start=0.15, stop=0.3, num=10),
    "Delta": logspace(start=-2, stop=1, num=10, base=10),
    "tau_m": (1 / 3.09e-4) * logspace(start=-1, stop=1, num=10, base=10),  # (s).,
}
MODELS = {"elastic": "PREM", "attenuation": "Resovsky", "transient": "reference", "viscous": "VM7"}
PARAMETERS_NAME = "parameters"
PARAMETERS_PATH = Path(".")


def compute_love_numbers_for_gins(
    love_numbers_for_gins_tabs: Optional[dict[str, ndarray]] = None,
    parameters_path: Path = PARAMETERS_PATH,
    parameters_file_name: str = PARAMETERS_NAME,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range of candidate
    physical models on alpha and Delta parameters.
    """

    if not love_numbers_for_gins_tabs:

        love_numbers_for_gins_tabs = LOVE_NUMBERS_FOR_GINS_TABS

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
        period_tab_per_degree={
            degree: love_numbers_for_gins_tabs["periods"]
            for degree in love_numbers_for_gins_tabs["degrees"]
        },
        parameters_to_invert_dictionary={
            r"\alpha^{MANTLE_0}": list(love_numbers_for_gins_tabs["alpha"]),
            r"\Delta^{MANTLE_0}": list(love_numbers_for_gins_tabs["Delta"]),
            r"\omega_{m-inf}^{MANTLE_0}": list(1 / love_numbers_for_gins_tabs["tau_m"]),
        },
        path=INTEGRATION_PATH,
    )
    save_base_model(
        obj=love_numbers_for_gins_tabs["periods"], name="periods_tab", path=INTEGRATION_PATH
    )


def load_love_numbers_for_gins(
    path: Path = INTEGRATION_PATH,
    models: Optional[dict[str, str]] = None,
    love_numbers_for_gins_tabs: Optional[dict[str, ndarray]] = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Gets already computed Love numbers of interest and their derivatives with respect to alpha,
    log10(Delta) and log10(tau_m).
    """

    if models is None:

        models = MODELS

    if love_numbers_for_gins_tabs is None:

        love_numbers_for_gins_tabs = LOVE_NUMBERS_FOR_GINS_TABS

    shape = (
        len(love_numbers_for_gins_tabs["alpha"]),
        len(love_numbers_for_gins_tabs["Delta"]),
        len(love_numbers_for_gins_tabs["tau_m"]),
        len(love_numbers_for_gins_tabs["degrees"]),
        len(love_numbers_for_gins_tabs["periods"]),
    )

    love_numbers = zeros(
        shape=shape,
        dtype=complex,
    )
    love_number_alpha_partials = zeros(
        shape=shape,
        dtype=complex,
    )
    love_number_log10_delta_partials = zeros(shape=shape, dtype=complex)
    love_number_log10_tau_m_partials = zeros(shape=shape, dtype=complex)

    for i_alpha in range(len(love_numbers_for_gins_tabs["alpha"])):

        for i_delta in range(len(love_numbers_for_gins_tabs["Delta"])):

            for i_tau_m in range(len(love_numbers_for_gins_tabs["tau_m"])):

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
                        parameters_to_invert=[
                            r"\alpha^{MANTLE_0}",
                            r"\Delta^{MANTLE_0}",
                            r"\omega_{m-inf}^{MANTLE_0}",
                        ],
                        invertible_parameter_tab=[
                            love_numbers_for_gins_tabs["alpha"][i_alpha],
                            love_numbers_for_gins_tabs["Delta"][i_delta],
                            1 / love_numbers_for_gins_tabs["tau_m"][i_tau_m],
                        ],
                    ),
                    path=path,
                )

                for i_degree, degree in love_numbers_for_gins_tabs["degrees"]:

                    love_numbers[i_alpha, i_delta, i_tau_m, i_degree] = (
                        model.love_numbers["real"][degree][
                            :,
                            BoundaryCondition.POTENTIAL.value,
                            Direction.POTENTIAL.value,
                        ]
                        + 1j
                        * model.love_numbers["imag"][degree][
                            :,
                            BoundaryCondition.POTENTIAL.value,
                            Direction.POTENTIAL.value,
                        ]
                    )
                    love_number_alpha_partials[i_alpha, i_delta, i_tau_m, i_degree] = (
                        model.love_number_partials["real"][r"\alpha^{MANTLE_0}"][degree][
                            :,
                            BoundaryCondition.POTENTIAL.value,
                            Direction.POTENTIAL.value,
                        ]
                        + 1j
                        * model.love_number_partials["imag"][r"\alpha^{MANTLE_0}"][degree][
                            :,
                            BoundaryCondition.POTENTIAL.value,
                            Direction.POTENTIAL.value,
                        ]
                    )
                    love_number_log10_delta_partials[i_alpha, i_delta, i_tau_m, i_degree] = (
                        log(10)
                        * love_numbers_for_gins_tabs["delta"][None, :, None, None, None]
                        * (
                            model.love_number_partials["real"][r"\Delta^{MANTLE_0}"][degree][
                                :,
                                BoundaryCondition.POTENTIAL.value,
                                Direction.POTENTIAL.value,
                            ]
                            + 1j
                            * model.love_number_partials["imag"][r"\Delta^{MANTLE_0}"][degree][
                                :,
                                BoundaryCondition.POTENTIAL.value,
                                Direction.POTENTIAL.value,
                            ]
                        )
                    )
                    love_number_log10_tau_m_partials[i_alpha, i_delta, i_tau_m, i_degree] = (
                        -log(10)
                        # Because of inverse change of variable.
                        * love_numbers_for_gins_tabs["tau_m"][None, :, None, None, None] ** 3
                        * (
                            model.love_number_partials["real"][r"\omega_{m-inf}^{MANTLE_0}"][
                                degree
                            ][
                                :,
                                BoundaryCondition.POTENTIAL.value,
                                Direction.POTENTIAL.value,
                            ]
                            + 1j
                            * model.love_number_partials["imag"][r"\omega_{m-inf}^{MANTLE_0}"][
                                degree
                            ][
                                :,
                                BoundaryCondition.POTENTIAL.value,
                                Direction.POTENTIAL.value,
                            ]
                        )
                    )

    model = load_solid_earth_numerical_model(
        name="PREM",
        path=TEST_ELASTIC_INTEGRATION_PATH,
    )

    return (
        array(
            object=[
                model.love_numbers["real"][degree][0][
                    BoundaryCondition.POTENTIAL.value,
                    Direction.POTENTIAL.value,
                ]
                for degree in love_numbers_for_gins_tabs["degrees"]
            ]
        ),
        love_numbers,
        love_number_alpha_partials,
        love_number_log10_delta_partials,
        love_number_log10_tau_m_partials,
    )
