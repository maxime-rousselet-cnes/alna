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
)
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap, subplots, suptitle, tight_layout
from numpy import array, diff, log10, meshgrid, ndarray, zeros

from alna import (
    COMPLEX_PARTS,
    SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR,
    load_solid_earth_numerical_model,
)
from integration_tests import (
    ALPHA_PERIOD_TAB,
    ALPHA_TAB,
    TEST_ALPHA_PARTIAL_INTEGRATION_PATH,
    TEST_ELASTIC_INTEGRATION_PATH,
    TEST_VISCOUS_INTEGRATION_PATH,
    VISCOUS_PERIOD_TAB,
    load_reference_love_numbers_for_validation,
)

DEFAULT_REFERENCE_LOVE_NUMBERS_PATH = Path("../../ViscoLove/EARTH_MODELS/PREM_ELASTIC")


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

    path.mkdir(exist_ok=True, parents=True)

    for file_format in ["svg", "png"]:

        figure.savefig(fname=path.joinpath(figure_title + "." + file_format), format=file_format)


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
    test_path: Path = TEST_VISCOUS_INTEGRATION_PATH,
    path: Path = TEST_FIGURES_PATH,
    period_tab: ndarray = VISCOUS_PERIOD_TAB,
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
            list(models.values()) + ["no_transient"]
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

    path.mkdir(exist_ok=True, parents=True)

    for file_format in ["svg", "png"]:

        figure.savefig(fname=path.joinpath(figure_title + "." + file_format), format=file_format)


def test_compare_plot_semi_analytical_partials_to_finite_differences(
    models: Optional[dict[str, str]] = None,
    test_path: Path = TEST_ALPHA_PARTIAL_INTEGRATION_PATH,
    periods_tab: ndarray = ALPHA_PERIOD_TAB,
    alpha_tab: ndarray = ALPHA_TAB,
    path: Path = TEST_FIGURES_PATH,
) -> None:
    """
    Generates a figure of 3 subplots as a function of degree and period, for h', l', and k'.
    Post-processes the needed data to do so.
    """

    if models is None:

        models = DEFAULT_MODELS

    love_numbers = zeros(shape=(len(periods_tab), 3, len(alpha_tab)))
    love_number_partials = zeros(shape=(len(periods_tab), 3, len(alpha_tab)))

    for i_alpha, alpha in enumerate(alpha_tab):

        solid_earth_numerical_model = load_solid_earth_numerical_model(
            name=SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR.join(
                list(models.values()) + [f"_alpha_{alpha:.3f}"]
            ),
            path=test_path,
        )
        love_numbers[:, :, i_alpha] = (
            solid_earth_numerical_model.love_numbers["real"][2][
                :, BoundaryCondition.POTENTIAL.value, :
            ]
            + 1j
            * solid_earth_numerical_model.love_numbers["imag"][2][
                :, BoundaryCondition.POTENTIAL.value, :
            ]
        )
        love_number_partials[:, :, i_alpha] = (
            solid_earth_numerical_model.love_number_partials["real"][2][
                :, BoundaryCondition.POTENTIAL.value, :
            ]
            + 1j
            * solid_earth_numerical_model.love_number_partials["imag"][2][
                :, BoundaryCondition.POTENTIAL.value, :
            ]
        )

    axes: Iterable[Iterable[Axes]]
    figure, axes = subplots(3, 2, figsize=(7, 12), sharex=True)
    figure_title = "Viscous Love numbers compared to elastic"
    suptitle(figure_title)

    for ax_line, label, direction in zip(axes, "hlk", Direction):

        for ax, part in zip(ax_line, COMPLEX_PARTS):

            for i_period, period in enumerate(periods_tab):

                (line,) = ax.plot(
                    alpha_tab,
                    (
                        love_number_partials[i_period, direction.value, :].real
                        if part == "real"
                        else love_number_partials[i_period, direction.value, :].imag
                    ),
                    label=f"{period:.1f} yr" if label == "h" and part == "real" else "",
                    linestyle="-",
                )
                ax.plot(
                    alpha_tab[:-1] + diff(a=alpha_tab) / 2,
                    diff(
                        a=(
                            love_numbers[i_period, direction.value, :].real
                            if part == "real"
                            else love_numbers[i_period, direction.value, :].imag
                        )
                    )
                    / diff(a=alpha_tab),
                    color=line.get_color(),
                    linestyle="--",
                )

            if part == "real":

                ax.set_ylabel(r"$\frac{\partial" + label + r"'_2}{\parial \alpha}$")
                ax.tick_params(labelbottom=False)

            if label == "k":

                ax.tick_params(labelbottom=True)
                ax.set_xlabel(r"$\alpha$")

                if part == "real":

                    ax.legend()

            elif label == "h":

                ax.set_title("Real part" if part == "real" else "Imaginary part")

    tight_layout()

    path.mkdir(exist_ok=True, parents=True)

    for file_format in ["svg", "png"]:

        figure.savefig(fname=path.joinpath(figure_title + "." + file_format), format=file_format)
