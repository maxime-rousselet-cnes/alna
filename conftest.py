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
        "--local_mode",
        action="store_true",
        default=False,
        help="Run tests in local mode.",
    )
    parser.addoption(
        "--n_periods",
        action="store",
        type=int,
        default=2,
        help="Number of periods to use in tests_integration.py.",
    )
    parser.addoption(
        "--n_parameter_values",
        action="store",
        type=int,
        default=2,
        help="Number of parameter values to use in tests_integration.py.",
    )


@fixture
def test_config(request: FixtureRequest):
    """
    For pytest to manage inline arguments.
    """

    return {
        "local_mode": request.config.getoption("--local_mode"),
        "n_periods": request.config.getoption("--n_periods"),
        "n_parameter_values": request.config.getoption("--n_parameter_values"),
    }
