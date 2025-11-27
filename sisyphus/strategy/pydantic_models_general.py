import json
from ast import literal_eval

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

# context method
MATERIAL_DESCRIPTION = """Processing-related identifiers for the material (e.g., 'annealed at 900°C', 'HPT processed'). Include only permanent material modifications: heat treatments, mechanical processing, surface treatments. Exclude testing conditions (testing environment), sample geometry, and measurement parameters. Processing parameters must be specific, if mutiple parameters given, separate them to different records. Return None if you are not sure or if the information is not available."""
REFERRED_DESCRIPTION = """This field should only be set to True for materials that are cited from other papers and not synthesized or prepared in the current paper. If a material, regardless of its prominence in the paper, is synthesized or prepared in the experimental section, it should not be marked as referred."""

class Material(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, e.g., 'Mn0.2CoCrNi'")
    composition_type: Literal['atomic', 'weight'] = Field(description="whether the composition is in atomic percent or weight percent")


class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, e.g., 'Mn0.2CoCrNi'. Return None if not available.")
    referred: bool = Field(description=REFERRED_DESCRIPTION)
    symbol_alias: Optional[str] = Field(description="The symbol or alias used to represent the material in paper, e.g., 'A', 'A-800'. If no symbol is given, return None.")
    processing_keywords: Optional[str] = Field(description=MATERIAL_DESCRIPTION)
    text: str = Field(description='Text spans from which the property is extracted, including essential information: material identity (e.g., composition, symbol, processing) and property. If extracted from table, include column header and relevant row. Be concise but informative.')



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
