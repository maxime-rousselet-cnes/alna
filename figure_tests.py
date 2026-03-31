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
from matplotlib.pyplot import get_cmap, subplots, suptitle, tight_layout
from numpy import array, log10, meshgrid, ndarray

from alna import SOLID_EARTH_NUMERICAL_MODEL_PART_NAMES_SEPARATOR, load_solid_earth_numerical_model
from integration_tests import (
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

            has_zero_value = (
                boundary_condition != BoundaryCondition.LOAD
            ) or direction == Direction.POTENTIAL
            ax.loglog(
                degrees_list[(0 if not has_zero_value else 1) :],
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
                    if not (i_n == 0 and has_zero_value)
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


def test_compare_plot_viscous_to_elastic(
    models: Optional[dict[str, str]] = None,
    elastic_test_path: Path = TEST_ELASTIC_INTEGRATION_PATH,
    test_path: Path = TEST_VISCOUS_INTEGRATION_PATH,
    path: Path = TEST_FIGURES_PATH,
    period_tab: ndarray = VISCOUS_PERIOD_TAB,
) -> None:
    """ """

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
    vmin, vmax, linthresh = -100, 100, 1
    cmap = get_cmap("seismic")
    yticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
    xticks = [1e1, 1e2, 1e3]

    for ax, label, direction in zip(axes, "hlk", Direction):

        im = ax.pcolormesh(
            degrees_mesh,
            periods_mesh,
            100 * all_data_to_plot[direction.value] * (1 if label == "h" else degrees_mesh),
            norm=SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax),
            cmap=cmap,
            shading="auto",
        )
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_box_aspect(1)
        cbar = figure.colorbar(im, ax=ax, ticks=[-100, -10, -1, 0, 1, 10, 100])
        cbar.set_label(("$" if label == "h" else "$n") + rf"\Delta {label}" + "'_n$ (%)")
        ax.set_ylabel("Period (yr)")
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{int(log10(t))}}}$" for t in yticks])
        ax.tick_params(labelbottom=False)

        if label == "k":

            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Degree")
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"$10^{{{int(log10(t))}}}$" for t in xticks])

    tight_layout()

    path.mkdir(exist_ok=True, parents=True)

    for file_format in ["svg", "png"]:

        figure.savefig(fname=path.joinpath(figure_title + "." + file_format), format=file_format)


test_compare_plot_viscous_to_elastic()
