import json
from ast import literal_eval

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

# context method
MATERIAL_DESCRIPTION = """Processing-related identifiers for the material (e.g., 'annealed at 900°C', 'HPT processed'). Include only permanent material modifications: heat treatments, mechanical processing, surface treatments. Exclude testing conditions, sample geometry, and measurement parameters. Processing parameters must be specific and not vague (e.g., do not use ranges). Return None if you are not sure or if the information is not available."""
REFERRED_DESCRIPTION = """This field should only be set to True for materials that are cited from other papers and not synthesized or prepared in the current paper. If a material, regardless of its prominence in the paper, is synthesized or prepared in the experimental section, it should not be marked as referred."""

class Material(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, e.g., 'Mn0.2CoCrNi'")
    composition_type: Literal['atomic', 'weight'] = Field(description="whether the composition is in atomic percent or weight percent")


class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, e.g., 'Mn0.2CoCrNi'. The composition field should only include information explicitly stated in the paper. Do not infer or make up any composition details that are not directly provided in the text.")
    referred: bool = Field(description=REFERRED_DESCRIPTION)
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
