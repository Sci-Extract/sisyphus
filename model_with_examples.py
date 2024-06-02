import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field


#### Configurable section #####
# example of filter, remember that customizable functions must conform with this schema
def regex_filter(doc: Document) -> bool:
    uptake_unit = re.compile(r'(cm3[/\s]g(-1)?)|(mmol[/\s]g(-1)?)', re.I)
    return bool(uptake_unit.search(doc.page_content))


# example of pydantic model
class ExtractUptake(BaseModel):
    """Extract uptake/adsorption properties from text"""

    mof_name: str = Field(..., description='the entity of gas uptaking, ')
    gas_type: str = Field(..., description='the gas used in adsorption/uptake')
    uptake: float = Field(
        ...,
        description='the quantity of the adsorption/uptake, e.g., 3',
    )
    uptake_unit: str = Field(
        ..., description='the unit of uptake, e.g., mmol/g'
    )
    temperature: Optional[float] = Field(
        None, description='process at which temperature'
    )
    temperature_unit: Optional[str] = Field(
        None, description='unit of temperature, e.g. "K"'
    )
    pressure: Optional[float] = Field(
        None, description='process at which pressure'
    )
    pressure_unit: Optional[str] = Field(
        None, description='unit of pressure, e.g., "KPa"'
    )


# NOTE: I recommand you to provide examples to get result match to provided.
tool_examples = [
    (
        'Example: The single-component isotherms revealed that [Cd2(dpip)2(DMF)(H2O)]·DMF·H2O adsorbs '
        '124.4/182.8 cm3 g−1 of C2H2, 76.8/120.0 cm3 g−1 of C2H4 '
        'at 298 and 273 K under 100 kPa, respectively.',
        [
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H2',
                uptake=124.4,
                uptake_unit='cm3/g',
                temperature=298,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H2',
                uptake=182.8,
                uptake_unit='cm3/g',
                temperature=273,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H4',
                uptake=76.8,
                uptake_unit='cm3/g',
                temperature=298,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H4',
                uptake=120.0,
                uptake_unit='cm3/g',
                temperature=273,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
        ],
    )
]
#### End #####
