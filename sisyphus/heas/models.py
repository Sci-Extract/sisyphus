from ast import literal_eval

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Literal
import json
import re

class Phase(BaseModel):
    phases: List[str] = Field(description='list of phases present in the material')
    test_env: Optional[str] = Field(description='other test parameters if avilable (not test method), be succinct, e.g., under plastic deformation')
    model_config = ConfigDict(extra="forbid")

class Strength(BaseModel):
    ys: Optional[str] = Field(description="Yield strength with unit")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain. If in percentage form, please add '%' sign, else return as decimal form. Example: 0.5 or 50%")
    temperature: Optional[str] = Field(description="Test temperature with unit, if not specified, return 'room temperature'")
    strain_rate: Optional[str] = Field(description="Strain rate with unit, e.g., 1e-3 s^-1")
    test_env: Optional[str] = Field(description="Other tensile/compressive test environments if available, be succinct. e.g., sample geometry, salt environment")
    test_type: Literal['tensile', 'compressive']
    model_config = ConfigDict(extra="forbid")

class GrainSize(BaseModel):
    grain_size: str = Field(description="Average grain size with unit, e.g., '10 μm'")
    model_config = ConfigDict(extra="forbid")

class MetaData(BaseModel):
    composition: str = Field(description='nominal material composition with basis marker, e.g., "AlCoCrFeNi2.5@at", "AlCoCrFeNi2.1@wt", "AlCoCrFeNi@at+Al2O3@wt[5%]"')
    model_config = ConfigDict(extra="forbid")

class Synthesis(BaseModel):
    """Synthesis information for a HEAs material.
    Note:
    - Use article synthesis section as only source of information.
    - Do not contain process related to test (e.g., tensile/compression test temperature) or characterization (e.g., XRD, SEM).
    """
    steps: str = Field(
        description=(
            "List of processing steps in chronological order as JSON objects. "
            "Example: ["
            '{"induction melting": {"power": "50 kW", "coil frequency": "10 kHz", "atmosphere": "argon", '
            '"pressure": "1 atm", "crucible material": "graphite", "liquid mixing time": "5 min", "number of remelts": "2"}}, '
            '{"annealing": {"temperature": "800 K", "duration": "1 h", "atmosphere": "argon"}}'
            "]. Return [] if no processing info is found; use empty strings for unknown values."
        )
    )
    # Forbid additional properties so JSON Schema has additionalProperties=false
    model_config = ConfigDict(extra="forbid")
    # @field_validator('steps', mode='after')
    # @classmethod
    # def load(cls, value: str):
    #     try:
    #         value = json.loads(value)
    #     except:
    #         value = literal_eval(value)
    #     return value

    @field_validator('steps', mode='after')
    @classmethod
    def load(cls, value):

        value = value.replace('\x00', '').replace('\u0000', '')

        value = re.sub(r'[\x00-\x1F\x7F]', '', value)

        try:
            return json.loads(value)
        except Exception:
            pass

        try:
            return literal_eval(value)
        except Exception:
            pass

        return value
        