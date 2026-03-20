"""
Simple independent functions.
"""

from pathlib import Path
from typing import Optional

from numpy import arange, concatenate


def generate_degrees_list(
    degree_thresholds: list[int],
    degree_steps: list[int],
    n_max: Optional[int] = None,
) -> list[int]:
    """
    Generates the list of degrees for which to compute Love numbers, given a list of thresholds and
    a list of steps.
    """

    if n_max:

        degree_thresholds = [threshold for threshold in degree_thresholds if threshold <= n_max]
        degree_thresholds += [n_max + degree_steps[-1]]
        degree_steps = degree_steps[: len(degree_thresholds) - 1]

    return concatenate(
        [
            arange(degree_thresholds[i], degree_thresholds[i + 1], degree_step, dtype=int)
            for i, degree_step in enumerate(degree_steps)
        ],
    ).tolist()


def get_periods(path: Path) -> list[float]:
    """
    Builds period list from directory names.
    """

    periods = [
        float(period_sub_path.name)
        for period_sub_path in path.iterdir()
        if period_sub_path.is_dir()
    ]
    periods.sort()

    return periods
