"""
To produce Love numbers of interest and their partial deriavtives for a candidate range of degrees,
periods and physical models.
"""

from itertools import product
from pathlib import Path
from typing import Optional

from base_models import MODELS, BoundaryCondition, Direction, load_base_model
from numpy import array, flip, log, logspace, ndarray, sort, unique, zeros

from .constants import (
    SOLID_EARTH_NUMERICAL_MODEL_NAME_FROM_INVERTIBLE_PARAMETERS_SEPARATOR,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
    TEST_ELASTIC_INTEGRATION_PATH,
)
from .integration_loops import (
    DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
    MultiParametersLoop,
    build_parameter_tab_parametrization,
    multi_parameter_integration,
)
from .load_solid_earth_model import load_solid_earth_numerical_model

LOG10_PERIOD_LOWER_BOUND = -2  # (yr).
LOG10_PERIOD_UPPER_BOUND = 4  # (yr).
TO_GET_INVERSE_DERIVATIVES = {r"\omega_{m-inf}^{MANTLE_0}": r"\tau_{m-inf}^{MANTLE_0}"}
GOT_INVERSE_DERIVATIVES = {v: k for k, v in TO_GET_INVERSE_DERIVATIVES.items()}
TO_GET_LOG_DERIVATIVES = [r"Q_\mu^{MANTLE_0}", r"\Delta^{MANTLE_0}", r"\tau_{m-inf}^{MANTLE_0}"]


def compute_love_numbers_for_gins(
    account: str = "",
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
        account=account,
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            degrees=degrees if degrees else [2],
            periods=logspace(
                start=LOG10_PERIOD_LOWER_BOUND,
                stop=LOG10_PERIOD_UPPER_BOUND,
                num=n_periods,
                base=10,
            ),
            parameters=build_parameter_tab_parametrization(n_parameter_values=n_parameter_values),
        ),
        models=models,
    )


def load_single_model_love_numbers_for_gins(
    file_path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    direction: Direction = Direction.POTENTIAL,
    boundary_condition: BoundaryCondition = BoundaryCondition.POTENTIAL,
) -> tuple[ndarray, ndarray, dict[str, ndarray]]:
    """
    Gets already computed Love numbers of interest and their derivatives with respect to alpha,
    log10(Q), log10(Delta) and log10(tau_m). Returns log frequencies, Love numbers, and Love number
    partials, degree axis and frequency axis following ascending order.
    """

    periods = array(object=load_base_model(name="periods_tab", path=file_path.parent), dtype=float)
    model = load_solid_earth_numerical_model(
        name=file_path.name,
        path=file_path.parent,
    )
    degrees = model.love_numbers["real"].keys()
    parameters = model.love_number_partials["real"].keys()
    love_numbers = zeros(shape=(len(degrees), len(periods)), dtype=complex)
    love_number_partials = {
        parameter: zeros(shape=(len(degrees), len(periods)), dtype=complex)
        for parameter in parameters
    }

    for i_degree, degree in enumerate(degrees):

        love_numbers[i_degree] = (
            model.love_numbers["real"][degree][:, boundary_condition.value, direction.value]
            + 1j * model.love_numbers["imag"][degree][:, boundary_condition.value, direction.value]
        )

        for parameter in parameters:

            love_number_partials[parameter][i_degree] = (
                model.love_number_partials["real"][parameter][degree][
                    :, boundary_condition.value, direction.value
                ]
                + 1j
                * model.love_number_partials["imag"][parameter][degree][
                    :, boundary_condition.value, direction.value
                ]
            )

    # Change of variables for inverse.
    for parameter in parameters:

        if parameter in TO_GET_INVERSE_DERIVATIVES.keys():

            parameter_value = float(file_path.name.split(parameter)[1][1:9])
            love_number_partials[TO_GET_INVERSE_DERIVATIVES[parameter]] = (
                -(parameter_value**2) * love_number_partials[parameter]
            )
            del love_number_partials[parameter]

    parameters = list(love_number_partials.keys())

    # Change of variables for log.
    for parameter in parameters:

        if parameter in TO_GET_LOG_DERIVATIVES:

            parameter_value = float(
                file_path.name.split(
                    parameter
                    if parameter not in GOT_INVERSE_DERIVATIVES
                    else GOT_INVERSE_DERIVATIVES[parameter]
                )[1][1:9]
            )

            if parameter in GOT_INVERSE_DERIVATIVES:

                parameter_value = 1 / parameter_value

            love_number_partials[r"\log_{10}" + parameter] = (
                log(10) * parameter_value * love_number_partials[parameter]
            )
            del love_number_partials[parameter]

    # Finally performs the period to frequency flip.
    return (
        flip(log(1 / periods)),
        flip(m=love_numbers, axis=-1),
        {
            parameter: flip(m=love_number_partials_for_parameter, axis=-1)
            for parameter, love_number_partials_for_parameter in love_number_partials.items()
        },
    )


def get_tabs_from_all_love_number_files(
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH.joinpath(DEFAULT_FOR_GINS_OUTPUT_DIRECTORY),
) -> dict[str, ndarray]:
    """
    Gets the parameter grid from the Love numbers folder.
    """

    parameters: list[dict] = []

    for file_path in path.glob("*"):

        parameter_name_and_values = file_path.name.split(
            SOLID_EARTH_NUMERICAL_MODEL_NAME_FROM_INVERTIBLE_PARAMETERS_SEPARATOR
        )[1:]
        parameters += [
            {
                parameter_name_and_value.split(".json")[0][:-9]: float(
                    parameter_name_and_value.split(".json")[0][-8:]
                )
                for parameter_name_and_value in parameter_name_and_values
            }
        ]

    parameter_tabs = {}

    for parameter_line in parameters:

        for parameter, value in parameter_line.items():

            if parameter in parameter_tabs:

                parameter_tabs[parameter] += [value]

            else:

                parameter_tabs[parameter] = [value]

    return {parameter: sort(unique(tab)) for parameter, tab in parameter_tabs.items()}


def modify(parameter: str, value: float) -> float:
    """
    TODO: describe.
    """

    return (
        (log(1 / value) if parameter in TO_GET_LOG_DERIVATIVES else 1 / value)
        if parameter in TO_GET_INVERSE_DERIVATIVES
        else (log(value) if parameter in TO_GET_LOG_DERIVATIVES else value)
    )


def load_love_numbers_for_gins(
    degrees: list[int] = [2],
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    directory: str = DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
    direction: Direction = Direction.POTENTIAL,
    boundary_condition: BoundaryCondition = BoundaryCondition.POTENTIAL,
) -> tuple[dict[str, ndarray], ndarray, ndarray, ndarray, dict[str, ndarray]]:
    """
    Gets already computed Love numbers of interest and their derivatives with respect to alpha,
    log10(Q), log10(Delta) and log10(tau_m). Returns parameter tabs after change of variable, log
    frequencies, elastic Love numbers, Love numbers, and Love number partials, every axis following
    ascending order.
    """

    love_numbers_for_gins_tabs = get_tabs_from_all_love_number_files(path=path.joinpath(directory))
    periods = array(
        object=load_base_model(name="periods_tab", path=path.joinpath(directory)), dtype=float
    )
    love_numbers = zeros(
        shape=tuple(
            [len(tab) for tab in love_numbers_for_gins_tabs.values()] + [len(degrees), len(periods)]
        ),
        dtype=complex,
    )
    love_number_partials = {}

    for iterators in product(*(range(len(tab)) for tab in love_numbers_for_gins_tabs.values())):

        file_finder = list(
            path.joinpath(directory).glob(
                "*"
                + "*".join(
                    (
                        f"{tab[iterator]:.2e}"
                        for iterator, tab in zip(iterators, love_numbers_for_gins_tabs.values())
                    )
                )
                + "*"
            )
        )

        if not file_finder:

            raise NameError

        _, love_numbers[iterators], love_number_partials_single_model = (
            load_single_model_love_numbers_for_gins(
                file_path=file_finder[0], direction=direction, boundary_condition=boundary_condition
            )
        )

        for (
            parameter,
            love_number_partials_for_parameter,
        ) in love_number_partials_single_model.items():

            if parameter not in love_number_partials:

                love_number_partials[parameter] = zeros(
                    shape=tuple(
                        [len(tab) for tab in love_numbers_for_gins_tabs.values()]
                        + [len(degrees), len(periods)]
                    ),
                    dtype=complex,
                )

            love_number_partials[parameter][iterators] = love_number_partials_for_parameter

    inverted_tabs = {}
    # Change of variables for inverse.
    for i_axis, parameter in enumerate(love_numbers_for_gins_tabs.keys()):

        if parameter in TO_GET_INVERSE_DERIVATIVES.keys():

            inverted_tabs[TO_GET_INVERSE_DERIVATIVES[parameter]] = 1 / flip(
                m=love_numbers_for_gins_tabs[parameter]
            )
            love_numbers = flip(m=love_numbers, axis=i_axis)

            for i_parameter in range(len(love_numbers_for_gins_tabs.keys())):

                love_number_partials[list(love_number_partials.keys())[i_parameter]] = flip(
                    m=love_number_partials[list(love_number_partials.keys())[i_parameter]],
                    axis=i_axis,
                )

        else:

            inverted_tabs[parameter] = love_numbers_for_gins_tabs[parameter]

    log_inverted_tabs = {}
    # Change of variables for log.
    for i_axis, parameter in enumerate(inverted_tabs.keys()):

        if parameter in TO_GET_LOG_DERIVATIVES:

            log_inverted_tabs[r"\log_{10}" + parameter] = log(inverted_tabs[parameter]) / log(10)

        else:

            log_inverted_tabs[parameter] = inverted_tabs[parameter]

    elastic = load_solid_earth_numerical_model(
        name="PREM",
        path=TEST_ELASTIC_INTEGRATION_PATH,
    ).love_numbers["real"]

    # Finally performs the period to frequency flip.
    return (
        log_inverted_tabs,
        flip(log(1 / periods)),
        array(
            object=[
                elastic[degree][0][
                    boundary_condition.value,
                    direction.value,
                ]
                for degree in degrees
            ]
        ),
        love_numbers,
        love_number_partials,
    )
