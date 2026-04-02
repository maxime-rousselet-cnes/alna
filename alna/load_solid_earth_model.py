"""
Defines a large function taking care of preprocessing when loading a solid Earth numerical model.
"""

from pathlib import Path
from typing import Optional

from base_models import load_base_model
from numpy import array, inf
from sympy import Symbol

from .solid_earth_model import SOLID_EARTH_NUMERICAL_MODELS_PATH, SolidEarthNumericalModel
from .sub_models import LayerModel


def load_solid_earth_numerical_model(
    name: str,
    path: Path = SOLID_EARTH_NUMERICAL_MODELS_PATH,
    force_transient: Optional[bool] = None,
) -> SolidEarthNumericalModel:
    """
    Loads a solid Earth numerical model and formats its expressions.
    """

    loaded_content = load_base_model(name=name, path=path)
    love_numbers: dict[str, dict[int, list[list[list[float]]]]] = loaded_content["love_numbers"]
    love_number_partials: dict[str, dict[str, dict[int, list[list[list[float]]]]]] = loaded_content[
        "love_number_partials"
    ]
    solid_earth_numerical_model = SolidEarthNumericalModel(
        name=loaded_content["name"],
        layer_models=[LayerModel() for _ in range(len(loaded_content["layer_models"]))],
        solid_earth_parameters=loaded_content["solid_earth_parameters"],
        units=loaded_content["units"],
        love_numbers={
            part: {
                n: array(object=love_numbers_tab)
                for n, love_numbers_tab in love_numbers_part.items()
            }
            for part, love_numbers_part in love_numbers.items()
        },
        love_number_partials={
            part: {
                parameter: {
                    n: array(object=love_number_partials_tab)
                    for n, love_number_partials_tab in partials.items()
                }
                for parameter, partials in love_number_partials_part.items()
            }
            for part, love_number_partials_part in love_number_partials.items()
        },
    )
    layer_model: dict[str, dict | str | float]

    for i_layer, layer_model in enumerate(loaded_content["layer_models"]):

        solid_earth_numerical_model.layer_models[i_layer].name = layer_model["name"]
        solid_earth_numerical_model.layer_models[i_layer].r_inf = layer_model["r_inf"]
        solid_earth_numerical_model.layer_models[i_layer].r_sup = layer_model["r_sup"]
        solid_earth_numerical_model.layer_models[i_layer].polynomials = {
            parameter: [inf] if "inf" in polynomial else polynomial
            for parameter, polynomial in layer_model["polynomials"].items()
        }
        solid_earth_numerical_model.layer_models[i_layer].parameter_symbols = {
            quantity: [Symbol(parameter) for parameter in polynomial]
            for quantity, polynomial in layer_model["parameter_symbols"].items()
        }

    if force_transient is not None:

        model = solid_earth_numerical_model.solid_earth_parameters.model
        model.component_parameters.transient_component = force_transient
        solid_earth_numerical_model.solid_earth_parameters.model = model

    return solid_earth_numerical_model

    return solid_earth_numerical_model
    return solid_earth_numerical_model
