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
    VARYING_MANTLE_PARAMETERS_TO_INVERT_BOUNDS,
    MultiParametersLoop,
    build_parameter_tab_parametrization,
    multi_parameter_integration,
)

VARYING_BIASING_MODELS = {
    "elastic": "PREM",
    "attenuation": "Resovsky",
    "transient": "unif_asth_non_asth",
    "viscous": "VM7_unif_lm",
}
VARYING_MANTLE_MODELS = {
    "elastic": "PREM",
    "attenuation": "Resovsky",
    "transient": "unif_asth_non_asth",
    "viscous": "VM7_lvz",
}
VARYING_BIASING_PARAMETERS_OUTPUT_DIRECTORY = "varying_biasing"
VARYING_MANTLE_PARAMETERS_OUTPUT_DIRECTORY = "varying_mantle"

# Exponentiation base if 3-rd parameter is present.
VARYING_BIASING_VARYING_MANTLE_PARAMETERS_TO_INVERT_BOUNDS = {
    r"\alpha^{ASTHENOSPHERE_0}": (0.16, 0.18),
    r"\Delta^{ASTHENOSPHERE_0}": (3, 15),
    r"\eta_m^{MANTLE-ASTHENOSPHERE_0}": (2e19, 3e19),
    r"\eta_m^{LOWER-MANTLE_0}": (1e21, 1e22),
}


def compute_love_numbers_for_varying_mantle(
    test_config: Config | dict[str, int | bool],
    degrees: Optional[list[int]] = None,
    varying_models: bool = False,
) -> None:
    """
    Computes Love numbers of interest and their partial deriavtives for a range of variations.
    """

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
                parameter_to_invert_bounds=VARYING_BIASING_VARYING_MANTLE_PARAMETERS_TO_INVERT_BOUNDS,
            ),
            output_directory=VARYING_BIASING_PARAMETERS_OUTPUT_DIRECTORY,
        ),
        models=VARYING_BIASING_MODELS,
    )

    if varying_models:

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
                    parameter_to_invert_bounds=VARYING_MANTLE_PARAMETERS_TO_INVERT_BOUNDS,
                ),
                output_directory=VARYING_MANTLE_PARAMETERS_OUTPUT_DIRECTORY,
            ),
            models=VARYING_MANTLE_MODELS,
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
    compute_love_numbers_for_varying_mantle(
        test_config={
            "account": args.account,
            "n_parameter_values": args.n_parameter_values,
            "n_periods": args.n_periods,
        },
        degrees=[2],
        varying_models=True,
    )
