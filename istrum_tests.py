"""
To answer:
    "What asthenosphere's thickness mimics a transient? For which signal wavelength?"
"""

from pathlib import Path

from base_models import DEFAULT_MODELS, BoundaryCondition, Direction, SolidEarthModelPart
from matplotlib.pyplot import legend, plot, show
from numpy import concatenate, linspace, log, logspace, ndarray, ones, sign, zeros
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d

from alna import (
    TEST_ELASTIC_INTEGRATION_PATH,
    SolidEarthNumericalModel,
    generate_degree_tab,
    load_solid_earth_numerical_model,
)
from base_tests import TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH, initialize_test

N_PERIODS_ISTRUM_INTEGRATION_TEST = 100
ISTRUM_PERIOD_TAB = logspace(
    -3, 5, num=N_PERIODS_ISTRUM_INTEGRATION_TEST, base=10
)  # (yr), from sub-daily to 100 kyr.

TEST_ISTRUM_INTEGRATION_PATH = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath("iSTRUM")
CLEANUP = False


def test_istrum_love_numbers(
    viscous_models: list[str] = [
        "iSTRUM_Maxwell_200km_Asthenosphere_" + str(i + 1) + "_10_19_Pas" for i in range(2)
    ],
    n_max: int = 1000,
    period_tab: ndarray = ISTRUM_PERIOD_TAB,
    elastic_test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    test_path: Path = TEST_ISTRUM_INTEGRATION_PATH,
) -> None:
    """
    Computes Love numbers for different viscous models to see when a low steady-state viscosity
    Asthenosphere can mimic a transient component in the upper mantle.
    """

    models = DEFAULT_MODELS if not CLEANUP else initialize_test(models=None, test_path=test_path)

    for viscous_model in viscous_models:  # + ["iSTRUM_Burgers"]:

        solid_earth_numerical_model: SolidEarthNumericalModel = load_solid_earth_numerical_model(
            name=models[SolidEarthModelPart.ELASTIC.value], path=elastic_test_path
        )
        solid_earth_numerical_model.merge_all(
            models=models | {SolidEarthModelPart.VISCOUS.value: viscous_model}
        )
        solid_earth_numerical_model.name = viscous_model
        components = solid_earth_numerical_model.solid_earth_parameters.model.component_parameters
        components.transient_component = False
        components.viscous_component = True
        solid_earth_numerical_model.solid_earth_parameters.model.component_parameters = components
        solid_earth_numerical_model.compute_love_numbers(
            period_tab_per_degree={
                degree: period_tab for degree in generate_degree_tab(n_max=n_max)
            },
            path=test_path,
            format_name=False,
        )


def load_istrum_love_numbers_and_convolve(
    period_tab: ndarray = ISTRUM_PERIOD_TAB,
    test_path: Path = TEST_ISTRUM_INTEGRATION_PATH,
    t_duration_years: float = 1e3,
    n_points: int = int(1e3),
    safety_factor: int = 10,
) -> None:
    """
    Test.
    """

    solid_earth_numerical_model: SolidEarthNumericalModel = load_solid_earth_numerical_model(
        name="iSTRUM_Maxwell_200km_Asthenosphere_10_19_Pas",
        path=test_path,
        # name="iSTRUM_Burgers"
    )
    elastic_solid_earth_numerical_model: SolidEarthNumericalModel = (
        load_solid_earth_numerical_model(
            name="PREM",
            path=TEST_ELASTIC_INTEGRATION_PATH,
        )
    )
    dates = linspace(start=0, stop=t_duration_years, num=n_points)  # (yr).
    signal = concatenate(
        [
            safety_factor * list(zeros(shape=dates.shape)),
            ones(shape=dates.shape),
            safety_factor * list(zeros(shape=dates.shape)),
        ]
    )
    periodic_signal = concatenate([-signal[::-1], signal])
    frequencies = fftfreq(n=(4 * safety_factor + 2) * n_points, d=dates[1] - dates[0])
    initial_frequencies = 1 / period_tab

    for degree in list(solid_earth_numerical_model.love_numbers["real"].keys())[:60:3]:

        if degree == 1 or degree not in elastic_solid_earth_numerical_model.love_numbers["real"]:

            continue

        frequency_dependent_love_numbers = (
            solid_earth_numerical_model.love_numbers["real"][degree]
            + solid_earth_numerical_model.love_numbers["imag"][degree] * 1j
        )
        frequency_dependent_h_prime_love_numbers = (
            frequency_dependent_love_numbers[
                :, BoundaryCondition.LOAD.value, Direction.VERTICAL.value
            ]
            / elastic_solid_earth_numerical_model.love_numbers["real"][degree][0][
                BoundaryCondition.LOAD.value, Direction.VERTICAL.value
            ]
        )

        ready_to_convolve_love_number = (
            interp1d(
                x=log(initial_frequencies),
                y=frequency_dependent_h_prime_love_numbers.real,
                fill_value=0,
                bounds_error=False,
            )(
                x=log(abs(frequencies)),
            )
            + sign(frequencies)
            * interp1d(
                x=log(initial_frequencies),
                y=frequency_dependent_h_prime_love_numbers.imag,
                fill_value=0,
                bounds_error=False,
            )(x=log(abs(frequencies)))
            * 1j
        )

        convolved_signal: ndarray = ifft(x=fft(periodic_signal) * ready_to_convolve_love_number)
        output = convolved_signal[
            (3 * safety_factor + 1) * n_points : (3 * safety_factor + 2) * n_points
        ].real

        plot(
            dates,
            output,
            label=degree,
        )

    legend()
    show()


if __name__ == "__main__":

    load_istrum_love_numbers_and_convolve()
