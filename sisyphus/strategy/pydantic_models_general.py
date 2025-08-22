import json
from ast import literal_eval

from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

# context method
class Material(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")

class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="nominal composition of the material, in at% e.g., 'Mn0.2CoCrNi'")
    description: Optional[str] = Field(description="Description of the material. Give the description based on the processing method, e.g., 'as-cast', 'annealed at 900C'. Do not contain test condition which describes the testing setup rather than the material itself, such as tested under 700C, under salted environment. If material is given with composition only, without any description, return None.")
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

