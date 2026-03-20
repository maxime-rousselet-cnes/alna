"""
Numerical constants.
"""

from pathlib import Path

from base_models import DATA_PATH, SOLID_EARTH_MODEL_PROFILES

### Solid Earth model descriptions.
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH = Path("../alna").joinpath(
    "solid_earth_model_profile_descriptions"
)
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH: dict[str, Path] = {
    model_part: SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH.joinpath(model_part)
    for model_part in SOLID_EARTH_MODEL_PROFILES
}

## Solid Earth numerical models.
SOLID_EARTH_NUMERICAL_MODELS_PATH = DATA_PATH.joinpath("solid_earth_numerical_models")

import numpy

# Universal Gravitationnal constant (m^3.kg^-1.s^-2).
G = 6.67430e-11

# s.y^-1
SECONDS_PER_YEAR = 365.25 * 86400


def years_to_seconds(period: float) -> float:
    """
    Time unit conversion.
    """

    return SECONDS_PER_YEAR * period


# For integration.
INITIAL_Y_VECTOR = numpy.array(
    object=[
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=complex,
)


# Other low level parameters.
LAYER_DECIMALS = 5
LAYER_DECIMALS = 5
