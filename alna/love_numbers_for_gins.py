"""
To produce Love numbers of interest and their partial deriavtives for a candidate range of degrees,
periods and physical models.
"""

from pathlib import Path
from typing import Optional

from base_models import MODELS, BoundaryCondition, Direction, load_base_model
from numpy import array, log, log10, logspace, ndarray, zeros

from .constants import SOLID_EARTH_NUMERICAL_MODELS_PATH, TEST_ELASTIC_INTEGRATION_PATH
from .integration_loops import (
    ALPHA_LOWER_BOUND,
    ALPHA_UPPER_BOUND,
    DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
    LOG10_DELTA_LOWER_BOUND,
    LOG10_DELTA_UPPER_BOUND,
    MultiParametersLoop,
    multi_parameter_integration,
)
from .load_solid_earth_model import load_solid_earth_numerical_model
from .parameters import (
    ComponentParameters,
    build_base_name,
    compose_name_with_invertible_parameters,
    format_name_function,
)

LOG10_PERIOD_LOWER_BOUND = -2  # (yr).
LOG10_PERIOD_UPPER_BOUND = 4  # (yr).
LOG10_TAU_M_OVER_REFERENCE_VALUE_LOWER_BOUND = -1  # (yr).
LOG10_TAU_M_OVER_REFERENCE_VALUE_UPPER_BOUND = 1  # (yr).
OMEGA_M_REFERENCE_VALUE = 3.09e-4


def parameters_for_gins(
    n_parameter_values: int = 2,
) -> dict[str, tuple[float, float, int] | tuple[float, float, int, float]]:
    """
    Generates parameter linspace arguments or logspace arguments.
    """

    return {
        r"\alpha^{MANTLE_0}": (ALPHA_LOWER_BOUND, ALPHA_UPPER_BOUND, n_parameter_values),
        r"\Delta^{MANTLE_0}": (
            LOG10_DELTA_LOWER_BOUND,
            LOG10_DELTA_UPPER_BOUND,
            n_parameter_values,
            10,
        ),
        r"\omega_{m-inf}^{MANTLE_0}": (
            log10(OMEGA_M_REFERENCE_VALUE) - LOG10_TAU_M_OVER_REFERENCE_VALUE_UPPER_BOUND,
            log10(OMEGA_M_REFERENCE_VALUE) - LOG10_TAU_M_OVER_REFERENCE_VALUE_LOWER_BOUND,
            n_parameter_values,
            10,
        ),
    }


def compute_love_numbers_for_gins(
    local_mode: bool = True,
    n_parameter_values: int = 2,
    n_periods: int = 2,
    degrees: Optional[list[int] | ndarray] = None,
    models: Optional[dict[str, str]] = None,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range of candidate
    physical models on alpha and Delta parameters.
    """

    if not models:

        models = MODELS

    multi_parameter_integration(
        local_mode=local_mode,
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            degrees=degrees if degrees else [2],
            periods=logspace(
                start=LOG10_PERIOD_LOWER_BOUND,
                stop=LOG10_PERIOD_UPPER_BOUND,
                num=n_periods,
                base=10,
            ),
            parameters=parameters_for_gins(n_parameter_values=n_parameter_values),
        ),
        models=models,
    )


def load_love_numbers_for_gins(
    n_parameter_values: int = 2,
    models: Optional[dict[str, str]] = None,
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    directory: str = DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
    love_numbers_for_gins_tabs: Optional[dict[str, ndarray]] = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Gets already computed Love numbers of interest and their derivatives with respect to alpha,
    log10(Delta) and log10(tau_m). Returns periods, Elastic Love numbers, Love numbers, and Love
    number partials.
    """

    if models is None:

        models = MODELS

    if love_numbers_for_gins_tabs is None:

        love_numbers_for_gins_tabs = parameters_for_gins(n_parameter_values=n_parameter_values)

    periods = array(
        object=load_base_model(name="periods_tab", path=path.joinpath(directory)), dtype=float
    )
    shape = (len(periods),)
    love_numbers = zeros(
        shape=shape,
        dtype=complex,
    )  # Overwritten later.
    love_number_partials = {
        parameter: zeros(
            shape=shape,
            dtype=complex,
        )
        for parameter in love_numbers_for_gins_tabs.keys()
    }  # Overwritten later.

    for i_alpha in range(len(love_numbers_for_gins_tabs[r"\alpha^{MANTLE_0}"])):

        for i_delta in range(len(love_numbers_for_gins_tabs[r"\Delta^{MANTLE_0}"])):

            for i_tau_m in range(len(love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"])):

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
                        invertible_parameters_tab=[
                            love_numbers_for_gins_tabs[r"\alpha^{MANTLE_0}"][i_alpha],
                            love_numbers_for_gins_tabs[r"\Delta^{MANTLE_0}"][i_delta],
                            love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"][i_tau_m],
                        ],
                    ),
                    path=path.joinpath(directory),
                )

                if len(shape) <= 1:

                    shape = tuple(
                        [len(tab) for tab in love_numbers_for_gins_tabs.values()]
                        + [len(model.love_numbers["real"].keys())]
                        + list(shape)
                    )
                    love_numbers = zeros(
                        shape=shape,
                        dtype=complex,
                    )
                    love_number_partials = {
                        parameter: zeros(
                            shape=shape,
                            dtype=complex,
                        )
                        for parameter in love_numbers_for_gins_tabs.keys()
                    }

                for i_degree, degree in enumerate(model.love_numbers["real"].keys()):

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
                    love_number_partials[r"\alpha^{MANTLE_0}"][
                        i_alpha, i_delta, i_tau_m, i_degree
                    ] = (
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
                    love_number_partials[r"\Delta^{MANTLE_0}"][
                        i_alpha, i_delta, i_tau_m, i_degree
                    ] = (
                        log(10)
                        * love_numbers_for_gins_tabs[r"\Delta^{MANTLE_0}"][
                            None, :, None, None, None
                        ]
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
                    love_number_partials[r"\omega_{m-inf}^{MANTLE_0}"][
                        i_alpha, i_delta, i_tau_m, i_degree
                    ] = (
                        -log(10)
                        # Because of inverse change of variable.
                        * love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"][
                            None, :, None, None, None
                        ]
                        ** (-3)
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
        periods,
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
        love_number_partials,
    )
