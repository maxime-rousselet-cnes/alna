"""
Command-line executable script for a single model run to allow for parallel computing.
"""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from base_models import load_base_model
from numpy import array, ndarray

from alna import (
    DEFAULT_PERIOD_TAB_PER_DEGREE_FILE_NAME,
    DEFAULT_PERIOD_TAB_PER_DEGREE_PATH,
    SOLID_EARTH_NUMERICAL_MODELS_PATH,
    load_solid_earth_numerical_model,
)

ELASTIC_PERIOD_TAB = array(object=[1.0], dtype=float)  # (yr).


def parse_parameter_values(items: list[str] | None) -> dict[str, float]:
    """
    Rearranges the parsed input into a convenient dictionary for the Solid Earth numerical model.
    """

    if not items:

        return {}

    if len(items) % 2 != 0:

        raise ArgumentTypeError(
            "--parameters must contain name/value pairs: NAME1 VALUE1 NAME2 VALUE2 ..."
        )

    parameters: dict[str, float] = {}

    for name, value in zip(items[0::2], items[1::2]):

        try:

            parameters[name] = float(value)

        except ValueError as exc:

            raise ArgumentTypeError(
                f"Invalid float value for parameter {name!r}: {value!r}"
            ) from exc

    return parameters


@dataclass
class Args(Namespace):
    """
    For convenient type readability.
    """

    parameters: dict[str, float]


def parse_general_args(parser: ArgumentParser) -> None:
    """
    Defines a common parsing function for command-line arguments working for both multi-job and
    single job command-line.
    """

    parser.add_argument("--name", required=True, help="Solid Earth numerical model.")
    parser.add_argument("--path", help="Path to the target directory.")
    parser.add_argument("--output_path", help="Path to the target directory.")
    parser.add_argument("--period_tab_per_degree", help="Input (.JSONL) file.")
    parser.add_argument("--period_tab_per_degree_path", help="Path to the target directory.")
    parser.add_argument("--force_transient", action="store_true", help="Force transient mode.")
    parser.add_argument("--force_viscous", action="store_true", help="Force viscous mode.")
    parser.add_argument(
        "--compute_partials", action="store_true", help="Compute partial derivatives."
    )
    parser.add_argument(
        "--force_not_transient", action="store_true", help="Force non-transient mode."
    )
    parser.add_argument("--force_not_viscous", action="store_true", help="Force non-viscous mode.")
    parser.add_argument(
        "--not_compute_partials", action="store_true", help="Do not compute partial derivatives."
    )
    parser.add_argument(
        "--not_format_name", action="store_true", help="Do not format name with parts and options."
    )


def parse_single_job_args() -> Namespace:
    """
    Defines a parsing function for command-line arguments.
    """

    parser = ArgumentParser()
    parse_general_args(parser=parser)
    parser.add_argument(
        "--parameters",
        nargs="*",
        default={},
        metavar=("NAME", "VALUE"),
        help="Optional parameter name/value pairs.",
    )
    args: Args = parser.parse_args()
    args.parameters = {} if not args.parameters else parse_parameter_values(items=args.parameters)

    return args


def love_numbers_single_run_main(
    solid_earth_numerical_models_path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    period_tab_per_degree: ndarray = ELASTIC_PERIOD_TAB,
    period_tab_per_degree_path: Path = DEFAULT_PERIOD_TAB_PER_DEGREE_PATH,
    period_tab_per_degree_file_name: str = DEFAULT_PERIOD_TAB_PER_DEGREE_FILE_NAME,
) -> None:
    """
    Single model run to allow for parallel computing.
    """

    args = parse_single_job_args()
    force_transient: Optional[bool] = None
    force_viscous: Optional[bool] = None

    if args.force_not_transient:

        force_transient = False

    if args.force_transient:

        force_transient = True

    if args.force_not_viscous:

        force_viscous = False

    if args.force_viscous:

        force_viscous = True

    if args.period_tab_per_degree_path:

        period_tab_per_degree_path = Path(args.period_tab_per_degree_path)

    solid_earth_numerical_model = load_solid_earth_numerical_model(
        name=args.name,
        path=solid_earth_numerical_models_path if not args.path else Path(args.path),
        force_transient=force_transient,
        force_viscous=force_viscous,
    )

    if args.not_compute_partials:

        solid_earth_numerical_model.solid_earth_parameters.compute_partials = False

    if args.compute_partials:

        solid_earth_numerical_model.solid_earth_parameters.compute_partials = True

    solid_earth_numerical_model.compute_love_numbers(
        period_tab_per_degree=(
            period_tab_per_degree
            if (not args.period_tab_per_degree and not args.period_tab_per_degree_path)
            else load_base_model(
                name=(
                    period_tab_per_degree_file_name
                    if not args.period_tab_per_degree
                    else args.period_tab_per_degree
                ),
                path=period_tab_per_degree_path,
            )
        ),
        parameters_to_invert=args.parameters,
        path=solid_earth_numerical_models_path if not args.output_path else Path(args.output_path),
        format_name=not args.not_format_name if args.not_format_name else True,
    )


if __name__ == "__main__":

    love_numbers_single_run_main()
