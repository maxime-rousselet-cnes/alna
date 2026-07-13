"""
Validation figures produced from tests. To run via pytest figure_tests.py.
"""

from pathlib import Path
from typing import Iterable, Optional

from base_models import (
    DEFAULT_MODELS,
    TEST_FIGURES_PATH,
    BoundaryCondition,
    Direction,
    SolidEarthModelPart,
    lagrange_order4,
    load_base_model,
)
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap, subplots, suptitle, tight_layout
from matplotlib.ticker import StrMethodFormatter
from numpy import (
    array,
    atan2,
    diff,
    divide,
    linspace,
    log,
    log10,
    logspace,
    meshgrid,
    ndarray,
    pi,
    zeros,
)

from alna import (
    COMPLEX_PARTS,
    DEFAULT_REFERENCE_LOVE_NUMBERS_PATH,
    MODELS,
    PARTIAL_PERIOD_TAB,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    TEST_ELASTIC_INTEGRATION_PATH,
    TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    ComponentParameters,
    build_base_name,
    compose_name_with_invertible_parameters,
    format_name_function,
    generate_parameter_lines,
    load_love_numbers_for_gins,
    load_reference_love_numbers_for_validation,
    load_solid_earth_numerical_model,
    parameters_for_gins,
    save_figure,
)


def test_compare_plot_to_elastic_reference(
    model: str = DEFAULT_MODELS[SolidEarthModelPart.ELASTIC.value],
    test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    reference_love_numbers_path: Path = DEFAULT_REFERENCE_LOVE_NUMBERS_PATH,
    path: Path = TEST_FIGURES_PATH,
) -> None:
    """
    Generates a figure of 3 subplots respectively for (h', l', k'), (h*, l*, k*) and (h, l, k).
    """

    degrees_list, reference_love_numbers = load_reference_love_numbers_for_validation(
        path=reference_love_numbers_path
    )
    solid_earth_numerical_model = load_solid_earth_numerical_model(name=model, path=test_path)

    axes: Iterable[Axes]
    figure, axes = subplots(3, figsize=(5, 7), sharex=True)
    figure_title = "Elastic Love numbers compared to reference"
    suptitle(figure_title)

    for ax, boundary_condition in zip(axes, BoundaryCondition):

        for label, direction in zip("hlk", Direction):

            ax.loglog(
                degrees_list[
                    (
                        0
                        if not (
                            (boundary_condition != BoundaryCondition.LOAD)
                            or direction == Direction.POTENTIAL
                        )
                        else 1
                    ) :
                ],
                [
                    100
                    * abs(
                        (
                            love_numbers[0, boundary_condition.value, direction.value]
                            - reference_love_numbers[i_n, boundary_condition.value, direction.value]
                        )
                        / reference_love_numbers[i_n, boundary_condition.value, direction.value]
                    )
                    for i_n, love_numbers in enumerate(
                        solid_earth_numerical_model.love_numbers["real"].values()
                    )
                    if not (
                        i_n == 0
                        and (
                            (boundary_condition != BoundaryCondition.LOAD)
                            or direction == Direction.POTENTIAL
                        )
                    )
                ],
                label=rf"$\Delta {label}"
                + (
                    "'"
                    if boundary_condition == BoundaryCondition.LOAD
                    else ("^*" if boundary_condition == BoundaryCondition.SHEAR else "")
                )
                + "_n$",
            )
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=5,
            width=1,
        )
        ax.set_ylabel("%")
        ax.legend(
            loc="upper center" if boundary_condition == BoundaryCondition.LOAD else "upper left",
            frameon=False,
        )

        if boundary_condition == BoundaryCondition.POTENTIAL:

            ax.set_xlabel("Degree")

    save_figure(figure=figure, figure_title=figure_title, path=path)


def sub_function_compare_plot_viscous_to_elastic(
    degrees_mesh: ndarray,
    periods_mesh: ndarray,
    axes: Iterable[Axes],
    all_data_to_plot: list[ndarray],
    figure: Figure,
) -> None:
    """
    Generates a figure of 3 subplots as a function of degree and period, for h', l', and k'.
    Takes the already post-processed data as an input.
    """

    yticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
    xticks = [1e1, 1e2, 1e3]

    for ax, label, direction in zip(axes, "hlk", Direction):

        im = ax.pcolormesh(
            degrees_mesh,
            periods_mesh,
            all_data_to_plot[direction.value] * (1 if label == "h" else degrees_mesh),
            # pylint: disable=unexpected-keyword-arg
            norm=SymLogNorm(linthresh=1, vmin=-100, vmax=100),
            cmap=get_cmap("seismic"),
            shading="auto",
        )
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_box_aspect(1)
        figure.colorbar(im, ax=ax, ticks=[-100, -10, -1, 0, 1, 10, 100]).set_label(
            ("$" if label == "h" else "$n") + rf"\Delta {label}" + "'_n$"
        )
        ax.set_ylabel("Period (yr)")
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{int(log10(t))}}}$" for t in yticks])
        ax.tick_params(labelbottom=False)

        if label == "k":

            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Degree")
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"$10^{{{int(log10(t))}}}$" for t in xticks])


def test_compare_plot_viscous_to_elastic(
    models: Optional[dict[str, str]] = None,
    elastic_test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath("viscous"),
    path: Path = TEST_FIGURES_PATH,
    period_tab: ndarray = array(
        object=load_base_model(
            name="periods_tab", path=TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH.joinpath("viscous")
        ),
        dtype=float,
    ),
) -> None:
    """
    Generates a figure of 3 subplots as a function of degree and period, for h', l', and k'.
    Post-processes the needed data to do so.
    """

    if models is None:

        models = DEFAULT_MODELS

    elastic_model = load_solid_earth_numerical_model(
        name=models[SolidEarthModelPart.ELASTIC.value], path=elastic_test_path
    )
    degrees = list(elastic_model.love_numbers["real"].keys())
    viscous_model = load_solid_earth_numerical_model(
        name=SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(
            (build_base_name(models=models), "no_transient")
        ),
        path=test_path,
    )
    all_data_to_plot = [
        array(
            object=[
                (
                    elastic_model.love_numbers["real"][n][0, 0, i_direction]
                    - viscous_model.love_numbers["real"][n][:, 0, i_direction]
                )
                for n in degrees[1:]
            ]
        ).T
        for i_direction in range(3)
    ]
    degrees_mesh, periods_mesh = meshgrid(degrees[1:], period_tab)
    axes: Iterable[Axes]
    figure, axes = subplots(3, figsize=(7, 12), sharex=True)
    figure_title = "Viscous Love numbers compared to elastic"
    suptitle(figure_title)
    sub_function_compare_plot_viscous_to_elastic(
        degrees_mesh=degrees_mesh,
        periods_mesh=periods_mesh,
        axes=axes,
        all_data_to_plot=all_data_to_plot,
        figure=figure,
    )
    tight_layout()
    save_figure(figure=figure, figure_title=figure_title, path=path)


def load_love_numbers_for_partials_plot(
    models: dict[str, str],
    test_path: Path,
    periods_tab: ndarray,
    parameter: str,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Load love numbers and their partials for all parameter values.
    """

    parameter_tab = array(
        object=[float(file.name.split(parameter)[1][1:-4]) for file in test_path.glob(parameter)],
        dtype=float,
    )
    love_numbers = zeros(shape=(len(periods_tab), 3, len(parameter_tab)), dtype=complex)
    love_number_partials = zeros(shape=(len(periods_tab), 3, len(parameter_tab)), dtype=complex)

    for i_parameter, parameter_value in enumerate(parameter_tab):

        solid_earth_numerical_model = load_solid_earth_numerical_model(
            name=compose_name_with_invertible_parameters(
                name=format_name_function(
                    name=build_base_name(models=models),
                    component_parameters=ComponentParameters(
                        viscous_component="eta" in parameter,
                        transient_component="alpha" in parameter or "Delta" in parameter,
                        bounded_attenuation_functions=True,
                    ),
                ),
                parameters_to_invert=[parameter],
                invertible_parameters_tab=[parameter_value],
            ),
            path=test_path,
        )
        love_numbers[:, :, i_parameter] = (
            solid_earth_numerical_model.love_numbers["real"][2][:, BoundaryCondition.LOAD.value, :]
            + 1j
            * solid_earth_numerical_model.love_numbers["imag"][2][
                :, BoundaryCondition.LOAD.value, :
            ]
        )
        love_number_partials[:, :, i_parameter] = (
            solid_earth_numerical_model.love_number_partials["real"][parameter][2][
                :, BoundaryCondition.LOAD.value, :
            ]
            + 1j
            * solid_earth_numerical_model.love_number_partials["imag"][parameter][2][
                :, BoundaryCondition.LOAD.value, :
            ]
        )

    return parameter_tab, love_numbers, love_number_partials


def plot_love_number_partials(
    axes: Iterable[Iterable[Axes]],
    love_numbers: ndarray,
    love_number_partials: ndarray,
    periods_tab: ndarray,
    parameter_tab: ndarray,
) -> None:
    """
    Plots love number partials against finite differences.
    """

    for ax_line, label, direction in zip(axes, "hlk", Direction):

        for ax, part in zip(ax_line, COMPLEX_PARTS):

            for i_period, period in enumerate(periods_tab):

                (line,) = ax.plot(
                    parameter_tab,
                    (
                        love_number_partials[i_period, direction.value, :].real
                        if part == "real"
                        else love_number_partials[i_period, direction.value, :].imag
                    ),
                    label=f"{period:.1f} yr" if label == "h" and part == "real" else "",
                    linestyle="-",
                    linewidth=2,
                )
                ax.plot(
                    parameter_tab[:-1] + diff(a=parameter_tab) / 2,
                    diff(
                        a=(
                            love_numbers[i_period, direction.value, :].real
                            if part == "real"
                            else love_numbers[i_period, direction.value, :].imag
                        )
                    )
                    / diff(parameter_tab),
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=5,
                )

                if part == "real":

                    ax.set_ylabel(r"$\frac{\partial " + label + r"'_2}{\partial p}$")
                    ax.tick_params(labelbottom=False)

                if label == "k":

                    ax.tick_params(labelbottom=True)
                    ax.set_xlabel(r"$p$")

                elif label == "h":

                    if part == "real":

                        ax.legend()

                    ax.set_title("Real part" if part == "real" else "Imaginary part")


def compare_plot_semi_analytical_partials_to_finite_differences(
    models: Optional[dict[str, str]] = None,
    parameter: str = r"\alpha^{MANTLE_0}",
    test_path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
) -> None:
    """
    Generates a figure of 3 subplots as a function of degree and period, for h', l', and k'.
    Post-processes the needed data to do so.
    """

    if models is None:

        models = DEFAULT_MODELS

    test_path = test_path.joinpath(parameter)
    periods_tab = array(object=load_base_model(name="periods_tab", path=test_path), dtype=float)
    parameter_tab, love_numbers, love_number_partials = load_love_numbers_for_partials_plot(
        models=models,
        test_path=test_path,
        periods_tab=periods_tab,
        parameter=parameter,
    )
    figure, axes = subplots(3, 2, figsize=(7, 12), sharex=True)
    suptitle("        $" + parameter + "$ partials")
    plot_love_number_partials(
        axes=axes,
        love_numbers=love_numbers,
        love_number_partials=love_number_partials,
        periods_tab=periods_tab,
        parameter_tab=parameter_tab,
    )
    tight_layout()
    save_figure(figure=figure, figure_title=parameter + " partials", path=TEST_FIGURES_PATH)


def test_compare_plot_semi_analytical_partials_to_finite_differences(
    models: Optional[dict[str, str]] = None,
) -> None:
    """
    Shows consistence of the methods between each other for an elastic parameter, a viscous
    parameter and a transient parameter.
    """

    compare_plot_semi_analytical_partials_to_finite_differences(
        models=models,
        parameter=r"\rho_0^{LOWER-MANTLE-1_0}",
    )
    compare_plot_semi_analytical_partials_to_finite_differences(
        models=models,
        parameter=r"\eta_m^{UPPER-MANTLE_0}",
    )
    compare_plot_semi_analytical_partials_to_finite_differences(
        models=models,
    )
    compare_plot_semi_analytical_partials_to_finite_differences(
        models=models,
        parameter=r"\Delta^{MANTLE_0}",
    )


def plot_interpolated_love_numbers_for_gins(
    love_numbers_to_plot: ndarray,
    omega_m_values_to_plot: ndarray,
    love_numbers_for_gins_tabs: dict[str, ndarray],
    period: float,
    figsize: tuple[float, float],
) -> Figure:
    """
    Plots interpolated love numbers for GINS.
    """

    axes: Iterable[Axes]
    figure, axes = subplots(
        len(omega_m_values_to_plot), 2, figsize=figsize, sharex=True, sharey=True
    )

    for i_tau_m, (omega_m, ax_line) in enumerate(zip(omega_m_values_to_plot, axes)):

        ax: Axes

        for ax, side in zip(ax_line, ["Modulus", "Phase"]):

            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.imshow(
                (
                    (
                        love_numbers_to_plot[:, :, i_tau_m].real ** 2
                        + love_numbers_to_plot[:, :, i_tau_m].imag ** 2
                    )
                    ** 0.5
                    if side == "Modulus"
                    else atan2(
                        love_numbers_to_plot[:, :, i_tau_m].imag,
                        love_numbers_to_plot[:, :, i_tau_m].real,
                    )
                    * period
                    / (2 * pi)
                    * (1 if period > 10 else 365)
                ),
                extent=[
                    love_numbers_for_gins_tabs[r"\alpha^{MANTLE_0}"].min(),
                    love_numbers_for_gins_tabs[r"\alpha^{MANTLE_0}"].max(),
                    love_numbers_for_gins_tabs[r"\Delta^{MANTLE_0}"].min(),
                    love_numbers_for_gins_tabs[r"\Delta^{MANTLE_0}"].max(),
                ],
                origin="lower",
                aspect="auto",
            )
            cbar = figure.colorbar(ax.images[0], ax=ax, orientation="vertical")
            cbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            cbar.set_label(
                (
                    ("        years" if period > 10 else "days")
                    if side == "Phase"
                    else r"          $|\frac{k_2(\omega)}{k_2^{el}}|$"
                ),
                fontsize=10 if side == "Phase" else 12,
                rotation=0,
            )

            ax.set_ylabel(r"$\Delta$")
            ax.set_xlabel(r"$\alpha$")
            ax.set_title(side + r": \tau_m = " + f"{1/omega_m:.4f} s")

    return figure


def plot_love_numbers_for_gins(
    love_numbers: ndarray,
    omega_m_values_to_plot: ndarray,
    love_numbers_for_gins_tabs: dict[str, ndarray],
    period: float,
    figsize: tuple[float, float],
) -> Figure:
    """
    Prepares a Modulus/Phase plot or k_2.
    """

    love_numbers_to_plot = zeros(
        shape=(
            len(love_numbers_for_gins_tabs[r"\alpha^{MANTLE_0}"]),
            len(love_numbers_for_gins_tabs[r"\Delta^{MANTLE_0}"]),
            len(love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"]),
        ),
        dtype=complex,
    )
    log_periods = array(object=log(love_numbers_for_gins_tabs["periods"]), dtype=float)

    # For degree 2, to interpolate on last component.
    for i_alpha, love_numbers_tab in enumerate(love_numbers[:, :, :, 0, :]):

        love_numbers_array: ndarray

        for i_delta, love_numbers_array in enumerate(love_numbers_tab):

            love_numbers_line: ndarray
            love_numbers_temp = zeros(
                shape=(len(love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"])), dtype=complex
            )

            for i_tau_m, love_numbers_line in enumerate(love_numbers_array):

                love_numbers_temp[i_tau_m] = (
                    lagrange_order4(
                        x=log_periods,
                        y=array(object=love_numbers_line.real, dtype=float),
                        new_x=array(object=[log(period)], dtype=float),
                    )[0]
                    + 1j
                    * lagrange_order4(
                        x=log_periods,
                        y=array(object=love_numbers_line.imag, dtype=float),
                        new_x=array(object=[log(period)], dtype=float),
                    )[0]
                )

            love_numbers_to_plot[i_alpha, i_delta, :] = lagrange_order4(
                x=log(love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"]),
                y=array(object=love_numbers_temp.real, dtype=float),
                new_x=log(omega_m_values_to_plot),
            ) + 1j * lagrange_order4(
                x=log(love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"]),
                y=array(object=love_numbers_temp.imag, dtype=float),
                new_x=log(omega_m_values_to_plot),
            )

    figure = plot_interpolated_love_numbers_for_gins(
        love_numbers_to_plot=love_numbers_to_plot,
        omega_m_values_to_plot=omega_m_values_to_plot,
        love_numbers_for_gins_tabs=love_numbers_for_gins_tabs,
        period=period,
        figsize=figsize,
    )
    tight_layout()

    return figure


def test_plot_k_2_love_numbers_for_gins(
    path: Path = TEST_SOLID_EARTH_NUMERICAL_MODEL_PATH,
    models: Optional[dict[str, str]] = None,
    n_parameter_values: int = 10,
    tau_values_to_plot_step: int = 1,
    periods_values_to_plot: list[float] = list(PARTIAL_PERIOD_TAB) + [433 / 365, 100.0],
) -> None:
    """
    Shows the GINS-ready Love numbers in 2D (alpha, delta) for real and imaginary parts.
    """

    love_numbers_for_gins_tabs = generate_parameter_lines(
        parameters=parameters_for_gins(n_parameter_values=n_parameter_values), write=False
    )

    for parameter, tab in love_numbers_for_gins_tabs.items():

        if len(tab) == 3:

            love_numbers_for_gins_tabs[parameter] = linspace(start=tab[0], stop=tab[1], num=tab[2])

        elif len(tab) == 4:

            love_numbers_for_gins_tabs[parameter] = logspace(
                start=tab[0], stop=tab[1], num=tab[2], base=tab[3]
            )

        else:

            love_numbers_for_gins_tabs[parameter] = tab

    if models is None:

        models = MODELS

    periods, elastic_love_numbers, love_numbers, _ = load_love_numbers_for_gins(
        path=path, models=models, love_numbers_for_gins_tabs=love_numbers_for_gins_tabs
    )  # (alpha, delta, tau_m, degrees, periods)
    love_numbers_for_gins_tabs["periods"] = periods
    omega_m_values_to_plot = love_numbers_for_gins_tabs[r"\omega_{m-inf}^{MANTLE_0}"][
        ::tau_values_to_plot_step
    ]

    for period in periods_values_to_plot:

        figure = plot_love_numbers_for_gins(
            love_numbers=divide(
                love_numbers,
                elastic_love_numbers[None, None, None, :, None],
                out=zeros(shape=love_numbers.shape, dtype=complex),
                where=elastic_love_numbers[None, None, None, :, None] != 0,
            ),
            omega_m_values_to_plot=array(object=omega_m_values_to_plot, dtype=float),
            love_numbers_for_gins_tabs=love_numbers_for_gins_tabs,
            period=period,
            figsize=(10, 5 * len(omega_m_values_to_plot)),
        )
        save_figure(
            figure=figure,
            figure_title="Love_numbers_for_gins_" + str(period) + "yr",
            path=TEST_FIGURES_PATH,
        )
