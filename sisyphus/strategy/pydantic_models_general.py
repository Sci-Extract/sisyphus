import json
from ast import literal_eval

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

# context method
MATERIAL_DESCRIPTION = """Processing-related identifiers for the material (e.g., 'annealed at 900°C', 'HPT processed'). Include only permanent material modifications: heat treatments, mechanical processing, surface treatments. Exclude testing conditions, sample geometry, and measurement parameters. Return None if you are not sured or the material is cited from other publications."""

class Material(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")


class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")
    refered: bool = Field(description="Indicate whether the material data is cited from other publications")
    symbol: Optional[str] = Field(description="The symbol used to represent the material in paper, e.g., 'A', 'A-800'. If no symbol is given, return None.")
    description: Optional[str] = Field(description=MATERIAL_DESCRIPTION)


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
