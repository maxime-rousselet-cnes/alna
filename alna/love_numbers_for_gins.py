"""
To produce Love numbers of interest and their partial deriavtives for a candidate range of degrees,
periods and physical models.
"""

from itertools import product
from pathlib import Path
from typing import Optional

from base_models import MODELS, BoundaryCondition, Direction, load_base_model
from numpy import array, flip, log, logspace, ndarray, zeros

from .constants import SOLID_EARTH_NUMERICAL_MODELS_PATH, TEST_ELASTIC_INTEGRATION_PATH
from .integration_loops import (
    DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
    MultiParametersLoop,
    build_parameter_tab_parametrization,
    multi_parameter_integration,
)
from .load_solid_earth_model import load_solid_earth_numerical_model
from .parameters import (
    ComponentParameters,
    build_base_name,
    compose_name_with_invertible_parameters,
    format_name_function,
    generate_parameter_lines,
)
from .solid_earth_model import SolidEarthNumericalModel

LOG10_PERIOD_LOWER_BOUND = -2  # (yr).
LOG10_PERIOD_UPPER_BOUND = 4  # (yr).


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


TO_GET_INVERSE_DERIVATIVES = {r"\omega_{m-inf}^{MANTLE_0}": r"\tau_{m-inf}^{MANTLE_0}"}
TO_GET_LOG_DERIVATIVES = [r"\Delta^{MANTLE_0}", r"\tau_{m-inf}^{MANTLE_0}"]


def load_love_numbers_for_gins(
    dummy_variable: int | SolidEarthNumericalModel = 2,
    models: Optional[dict[str, str]] = None,
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    directory: str = DEFAULT_FOR_GINS_OUTPUT_DIRECTORY,
) -> tuple[dict[str, ndarray], ndarray, ndarray, ndarray, dict[str, ndarray]]:
    """
    Gets already computed Love numbers of interest and their derivatives with respect to alpha, Q_mu
    log10(Delta) and log10(tau_m). Returns parameter tabs after change of variable, log frequencies,
    elastic Love numbers, Love numbers, and Love number partials, every axis following ascending
    order.
    """

    if models is None:

        models = MODELS

    love_numbers_for_gins_tabs = generate_parameter_lines(
        parameters=build_parameter_tab_parametrization(n_parameter_values=dummy_variable),
        write=False,
    )

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

    print(love_numbers_for_gins_tabs)

    for iterators in product(*(range(len(tab)) for tab in love_numbers_for_gins_tabs.values())):

        # Loads data for a point in parameter space.
        name = compose_name_with_invertible_parameters(
            name=format_name_function(
                name=build_base_name(models=models),
                component_parameters=ComponentParameters(
                    viscous_component=True,
                    transient_component=True,
                    bounded_attenuation_functions=True,
                ),
            ),
            parameters_to_invert=love_numbers_for_gins_tabs.keys(),
            invertible_parameters_tab=[
                love_numbers_for_gins_tabs[parameter][iterator]
                for parameter, iterator in zip(love_numbers_for_gins_tabs.keys(), iterators)
            ],
        )
        print(path.joinpath(directory))
        print(name)
        print()
        dummy_variable = load_solid_earth_numerical_model(
            name=list(path.joinpath(directory).glob("*" + name + "*"))[0].name,
            path=path.joinpath(directory),
        )

        # Adapts to the actual size of data.
        if len(shape) <= 1:

            shape = tuple(
                [len(tab) for tab in love_numbers_for_gins_tabs.values()]
                + [len(dummy_variable.love_numbers["real"].keys())]
                + list(shape)  # len(periods)
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

        for i_degree, degree in enumerate(dummy_variable.love_numbers["real"].keys()):

            love_numbers[iterators + (i_degree,)] = (
                dummy_variable.love_numbers["real"][degree][
                    :,
                    BoundaryCondition.POTENTIAL.value,
                    Direction.POTENTIAL.value,
                ]
                + 1j
                * dummy_variable.love_numbers["imag"][degree][
                    :,
                    BoundaryCondition.POTENTIAL.value,
                    Direction.POTENTIAL.value,
                ]
            )

            for parameter in love_numbers_for_gins_tabs.keys():

                love_number_partials[parameter][iterators + (i_degree,)] = (
                    dummy_variable.love_number_partials["real"][parameter][degree][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        Direction.POTENTIAL.value,
                    ]
                    + 1j
                    * dummy_variable.love_number_partials["imag"][parameter][degree][
                        :,
                        BoundaryCondition.POTENTIAL.value,
                        Direction.POTENTIAL.value,
                    ]
                )

    inverted_tabs = {}
    # Change of variables for inverse.
    for i_axis, (parameter, parameter_values) in enumerate(love_numbers_for_gins_tabs.items()):

        if parameter in TO_GET_INVERSE_DERIVATIVES.keys():

            idx = [None] * len(shape)
            idx[i_axis] = slice(None)
            love_number_partials[TO_GET_INVERSE_DERIVATIVES[parameter]] = (
                -parameter_values[tuple(idx)] ** 2 * love_number_partials[parameter]
            )
            inverted_tabs[TO_GET_INVERSE_DERIVATIVES[parameter]] = 1 / flip(
                m=love_numbers_for_gins_tabs[parameter]
            )
            love_numbers = flip(m=love_numbers, axis=i_axis)

            for parameter in love_numbers_for_gins_tabs.keys():

                love_number_partials[parameter] = flip(
                    m=love_number_partials[parameter], axis=i_axis
                )

            del love_number_partials[parameter]

        else:

            inverted_tabs[parameter] = love_numbers_for_gins_tabs[parameter]

    log_inverted_tabs = {}
    # Change of variables for log.
    for i_axis, (parameter, parameter_values) in enumerate(inverted_tabs.items()):

        if parameter in TO_GET_LOG_DERIVATIVES:

            idx = [None] * len(shape)
            idx[i_axis] = slice(None)
            love_number_partials[r"\log_{10}" + parameter] = (
                log(10) * parameter_values[tuple(idx)] * love_number_partials[parameter]
            )
            log_inverted_tabs[r"\log_{10}" + parameter] = log(inverted_tabs[parameter]) / log(10)
            del love_number_partials[parameter]

        else:

            log_inverted_tabs[parameter] = inverted_tabs[parameter]

    # Finally performs the period to frequency flip.
    return (
        log_inverted_tabs,
        flip(log(1 / periods)),
        array(
            object=[
                load_solid_earth_numerical_model(
                    name="PREM",
                    path=TEST_ELASTIC_INTEGRATION_PATH,
                ).love_numbers["real"][degree][0][
                    BoundaryCondition.POTENTIAL.value,
                    Direction.POTENTIAL.value,
                ]
                for degree in dummy_variable.love_numbers["real"].keys()
            ]
        ),
        flip(m=love_numbers, axis=-1),
        {
            parameter: flip(m=love_number_partials_for_parameter, axis=-1)
            for parameter, love_number_partials_for_parameter in love_number_partials.items()
        },
    )
