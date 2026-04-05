"""
Defines all parameter classes.
"""

from typing import Optional

from base_models import EARTH_RADIUS
from pydantic import BaseModel


class ComponentParameters(BaseModel):
    """
    Options to choose rheology components to include in the model.
    Elasticity is not optional.
    """

    viscous_component: bool = False
    transient_component: bool = False
    # Unbounded attenuation functions can be used in the transient component to check consistency
    # with specific short-term Love numbers litterature. However, they produce non-physical values
    # at long-term.
    bounded_attenuation_functions: bool = False


DEFAULT_COMPONENT_PARAMETERS = ComponentParameters()


class StructureParameters(BaseModel):
    """
    Defines the solid Earth model parameters usefull for y_i system integration.
    """

    dynamic_term: bool = True  # Whether to use omega^2 terms in the y_i system or not.
    # Number of layers under boundaries. If they are None: Automatic detection using elasticity
    # model layer names.
    # Number of layers under the Inner-Core Boundary.
    i_layer_icb: Optional[int] = None  # Should be >= 0.
    # Number of total layers under the Mantle-Core Boundary.
    i_layer_cmb: Optional[int] = None  # Should be >= i_layer_icb.
    asymptotic_compressibility: bool = False
    drho_dx_epsilon: float = (
        1.0e-10  # Limit under which incompressibility is assumed (for Lithosphere mainly).
    )


DEFAULT_STRUCTURE_PARAMETERS = StructureParameters()


class SolidEarthModelParameters(BaseModel):
    """
    Parameterizes the solid Earth model.
    """

    component_parameters: ComponentParameters = DEFAULT_COMPONENT_PARAMETERS
    # Whether to use 'optional_crust_values' values specified in the component profile description
    # file or not.
    # Usefull to easily switch from ocenanic to continental crust parameters.
    optional_crust_values: Optional[bool] = None
    # Length unit (m). Will be set to the default constant if None.
    radius_unit: Optional[float] = None
    structure_parameters: StructureParameters = DEFAULT_STRUCTURE_PARAMETERS

    def __init_subclass__(cls, **kwargs):

        return super().__init_subclass__(**kwargs)

    def __init__(
        self,
        component_parameters: ComponentParameters = DEFAULT_COMPONENT_PARAMETERS,
        optional_crust_values: Optional[bool] = None,
        radius_unit: Optional[float] = None,
        structure_parameters: StructureParameters = (DEFAULT_STRUCTURE_PARAMETERS),
    ):

        super().__init__()

        self.component_parameters = (
            component_parameters
            if isinstance(component_parameters, ComponentParameters)
            else ComponentParameters(**component_parameters)
        )
        self.optional_crust_values = (
            False if optional_crust_values is None else optional_crust_values
        )
        self.radius_unit = EARTH_RADIUS if radius_unit is None else radius_unit
        self.structure_parameters = (
            structure_parameters
            if isinstance(structure_parameters, StructureParameters)
            else StructureParameters(**structure_parameters)
        )


DEFAULT_SOLID_EARTH_MODEL_PARAMETERS = SolidEarthModelParameters()


class IntegrationParameters(BaseModel):
    """
    Describes the parameters necessary for the numerical integration of the y_i system.
    """

    high_degrees_radius_sensibility: (
        float  # Integrates starting whenever x**n > high_degrees_radius_sensibility.
    ) = 1.0e-4
    minimal_radius: float = 1.0e3  # r ~= 0 km exact definition (m).
    minimal_layer_radius_factor: float = 1.0
    atol: float = 1.0e-14  # The solver keeps the local error estimates under atol + rtol * abs(yr).
    rtol: float = 1.0e-10  # See atol parameter description.


DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS = IntegrationParameters()


class SolidEarthOptionParameters(BaseModel):
    """
    Parameters for optional computations
    """

    model_id: Optional[str] = None
    save: bool = True
    overwrite_model: bool = False


DEFAULT_SOLID_EARTH_OPTION_PARAMETERS = SolidEarthOptionParameters()


class SolidEarthParameters(BaseModel):
    """
    Defines all solid Earth algorithm parameters.
    """

    model: SolidEarthModelParameters = DEFAULT_SOLID_EARTH_MODEL_PARAMETERS
    n_max: Optional[int] = None
    integration_parameters: IntegrationParameters = DEFAULT_SOLID_EARTH_INTEGRATION_PARAMETERS
    options: SolidEarthOptionParameters = DEFAULT_SOLID_EARTH_OPTION_PARAMETERS


DEFAULT_SOLID_EARTH_PARAMETERS = SolidEarthParameters()


# List ofthe possible non-elastic rheologies.
ALL_COMPONENT_PARAMETERS: list[ComponentParameters] = [
    ComponentParameters(
        viscous_component=True,
        transient_component=True,
        bounded_attenuation_functions=True,
    ),
    ComponentParameters(
        viscous_component=False,
        transient_component=True,
        bounded_attenuation_functions=True,
    ),
    ComponentParameters(
        viscous_component=True,
        transient_component=False,
        bounded_attenuation_functions=False,
    ),
]

# Elastic rheology.
ELASTIC_COMPONENT_PARAMETERS = ComponentParameters(
    viscous_component=False,
    transient_component=False,
    bounded_attenuation_functions=False,
)
