import json
from ast import literal_eval

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

# context method
MATERIAL_DESCRIPTION = """Description of the material, which help you identify the material's processing steps later. Give the description based on the processing method, e.g., 'as-cast', 'annealed at 900C'. INCLUDE: Processing steps, heat treatments, mechanical working, surface treatments that permanently alter the material.
EXCLUDE: Testing conditions (e.g., 'tested at 700°C', 'measured in salt water'), sample geometry (e.g., 'dog-bone shaped'), or measurement parameters that describe the experimental setup rather than the material's intrinsic state.return None."""

class Material(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")


class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")
    description: Optional[str] = Field(description=MATERIAL_DESCRIPTION)
    refered: bool = Field(description="Indicate whether the material data is cited from other publications for comparison purposes.")


class Processing(Material):
    """Processing route for a material
    Return field steps as '[]' if you cannot find any. For fields with unknown value, filled with empty string"""
    steps: str = Field(description="""List of processing steps in chronological order, form them as json object. For example: [{"induction melting": {"temperature": "1500 K"}}, {"annealed": {"temperature": "800 K", "duration": "1h"}}]""")

    @field_validator('steps', mode='after')
    @classmethod
    def load(cls, value: str):
        try:
            value = json.loads(value)
        except:
            value = literal_eval(value)
        return value


class SafeDumpProcessing(Material):
    """Safe dump"""
    model_config = ConfigDict(from_attributes=True)
    steps: list[dict]


# isolated method
class MaterialWithSymbol(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")
    symbol: Optional[str] = Field(description="The symbol used to represent the material in paper, e.g., 'A', 'A-800'. If no symbol is given, return None.")


class ProcessingWithSymbol(MaterialWithSymbol):
    steps: str = Field(description="""List of processing steps in chronological order, form them as json object. For example: [{"induction melting": {"temperature": "1500 K"}}, {"annealed": {"temperature": "800 K", "duration": "1h"}}]""")

    @field_validator('steps', mode='after')
    @classmethod
    def load(cls, value: str):
        try:
            value = json.loads(value)
        except:
            value = literal_eval(value)
        return value


class SafeDumpProcessingWithSymbol(MaterialWithSymbol):
    """Safe dump"""
    model_config = ConfigDict(from_attributes=True)
    steps: list[dict]
