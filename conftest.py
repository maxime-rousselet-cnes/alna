"""
Tests configuration to handle optional inline arguments.
"""

from argparse import Namespace

from pytest import FixtureRequest, fixture


def pytest_addoption(parser: Namespace):
    """
    Optional inline arguments.
    """

    parser.addoption(
        "--account",
        action="store",
        type=str,
        help="Run tests in local mode if account=''.",
        default="grgs",
    )
    parser.addoption(
        "--n_periods",
        action="store",
        type=int,
        help="Number of periods to use in tests_integration.py.",
        default=2,
    )
    parser.addoption(
        "--n_parameter_values",
        action="store",
        type=int,
        help="Number of parameter values to use in tests_integration.py.",
        default=2,
    )


@fixture
def test_config(request: FixtureRequest):
    """
    For pytest to manage inline arguments.
    """

    return {
        "account": request.config.getoption("--account"),
        "n_periods": request.config.getoption("--n_periods"),
        "n_parameter_values": request.config.getoption("--n_parameter_values"),
    }
