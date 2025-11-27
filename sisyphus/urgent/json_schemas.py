# json templates
phase_template = """
{
    "metadata": {
        "composition": "%s",
        "label": "%s",
        "processing_kw": %s
    },
    "properties": {
        "phase": {
            "phases": [%s]
        }
    }
}
"""

strength_template = """
{
    "metadata": {
        "composition": "%s",
        "label": "%s",
        "processing_kw": %s
    },
    "properties": {
        "strength": {
            "ys": %d,
            "uts": %d,
            "strain": %d,
            "temperature": %d,
            "test_type": Literal['tensile', 'compression'] 
        }
    }
}
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class Phase(BaseModel):
    phases: List[str] = Field(description='list of phases present in the material')
    test_env: Optional[str] = Field(description='other test parameters if avilable (not test method), be succinct, e.g., under plastic deformation')

class Strength(BaseModel):
    """Tensile/Compressive test results"""
    ys: Optional[str] = Field(description="Yield strength with unit")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain. If in percentage form, please add '%' sign, else return as decimal form. Example: 0.5 or 50%")
    temperature: Optional[str] = Field(description="Test temperature with unit, if not specified, return 'room temperature'")
    strain_rate: Optional[str] = Field(description="Strain rate with unit, e.g., 1e-3 s^-1")
    test_env: Optional[str] = Field(description="Other tensile/compressive test environments if available, be succinct. e.g., sample geometry, salt environment")
    test_type: Literal['tensile', 'compressive']

class GrainSize(BaseModel):
    grain_size: str = Field(description="Average grain size with unit, e.g., '10 μm'")

class MetaData(BaseModel):
    composition: str = Field(description='material composition as a string')
    label: Optional[str] = Field(description='label for the sample, normally abbreviation like A1, B2, etc.')
    processing_kw: Optional[list[str]] = Field(description='processing keywords and parameters')

class PhaseRecord(BaseModel):
    metadata: MetaData
    phase: Phase
    referred: bool = Field(description="True if this information is explicitly cited from or attributed to another paper/source in the text (e.g., references like 'Smith et al. reported...', 'according to [5]', 'as shown in previous work [12]'). False if it represents the authors' own original findings or claims without citation."
)

class StrengthRecord(BaseModel):
    metadata: MetaData
    strength: Strength
    referred: bool = Field(description="True if this information is explicitly cited from or attributed to another paper/source in the text (e.g., references like 'Smith et al. reported...', 'according to [5]', 'as shown in previous work [12]'). False if it represents the authors' own original findings or claims without citation."
)

class GrainSizeRecord(BaseModel):
    metadata: MetaData
    grain_size: GrainSize
    referred: bool = Field(description="True if this information is explicitly cited from or attributed to another paper/source in the text (e.g., references like 'Smith et al. reported...', 'according to [5]', 'as shown in previous work [12]'). False if it represents the authors' own original findings or claims without citation."
)

class PhaseRecords(BaseModel):
    records: Optional[List[PhaseRecord]]

class StrengthRecords(BaseModel):
    records: Optional[List[StrengthRecord]]

class GrainSizeRecords(BaseModel):
    records: Optional[List[GrainSizeRecord]]

 