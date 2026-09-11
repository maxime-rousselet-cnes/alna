"""
Tests the consistency of integration. To test via pytest integration_tests.py.
"""

from argparse import ArgumentParser, Namespace
from typing import Optional

from numpy import logspace
from pytest import Config

from alna import (
    LOG10_PERIOD_LOWER_BOUND,
    LOG10_PERIOD_UPPER_BOUND,
    MultiParametersLoop,
    build_parameter_tab_parametrization,
    multi_parameter_integration,
)

VARYING_UPPER_MANTLE_MODELS = {
    "elastic": "PREM",
    "attenuation": "Resovsky",
    "transient": "unif_asth_non_asth",
    "viscous": "VM7_unif_lm",
}
VARYING_UPPER_MANTLE_OUTPUT_DIRECTORY = "varying_non_asth"

# Exponentiation base if 3-rd parameter is present.
VARYING_UPPER_MANTLE_PARAMETERS_TO_INVERT_BOUNDS = {
    r"\Delta^{NON-ASTH-MANTLE_0}": (-2, 1, 10.0),
}


def compute_love_numbers_for_varying_upper_mantle(
    test_config: Config | dict[str, int | bool],
    degrees: Optional[list[int]] = None,
    models: Optional[dict[str, str]] = None,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range
    of asthenospheric variations.
    """

    if not models:

        models = VARYING_UPPER_MANTLE_MODELS

    multi_parameter_integration(
        account=test_config["account"],
        multi_parameter_love_numbers_loop=MultiParametersLoop(
            degrees=degrees if degrees else [2],
            periods=logspace(
                start=LOG10_PERIOD_LOWER_BOUND,
                stop=LOG10_PERIOD_UPPER_BOUND,
                num=test_config["n_periods"],
                base=10,
            ),
            parameters=build_parameter_tab_parametrization(
                n_parameter_values=test_config["n_parameter_values"],
                parameter_to_invert_bounds=VARYING_UPPER_MANTLE_PARAMETERS_TO_INVERT_BOUNDS,
            ),
            output_directory=VARYING_UPPER_MANTLE_OUTPUT_DIRECTORY,
        ),
        models=models,
    )


def parse_args() -> Namespace:
    """
    Parses the same 3 optional parameters for direct Python execution.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--account",
        action="store",
        type=str,
        help="Run the integration in local mode if account=''.",
        default="grgs",
    )
    parser.add_argument(
        "--n_parameter_values",
        type=int,
        help="Number of parameter values to test for GINS-ready Love numbers.",
        default=2,
    )
    parser.add_argument(
        "--n_periods",
        type=int,
        help="Number of periods to integrate the Love numbers at.",
        default=2,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    compute_love_numbers_for_varying_upper_mantle(
        test_config={
            "account": args.account,
            "n_parameter_values": args.n_parameter_values,
            "n_periods": args.n_periods,
        },
        degrees=[2],
        models=VARYING_UPPER_MANTLE_MODELS,
    )
