"""
The class describes the radial quantities of a solid earth layer.
"""

from typing import Optional

import numpy
from base_models import adaptive_runge_kutta_45
from numpy import inf, ones, shape
from pydantic import BaseModel
from scipy.interpolate import make_lsq_spline, splev

from .parameters import SolidEarthNumericalParameters


class ModelLayer(BaseModel):
    """
    Defines a numerical model's layer.
    """

    name: Optional[str] = None
    x_inf: float = 0.0
    x_sup: float = 0.0
    polynomial_coefficients_per_variable: dict[str, list[float]] = {}
