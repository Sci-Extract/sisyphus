from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from sisyphus.heas.models import Strength, GrainSize, Phase, Synthesis


class MetaData(BaseModel):
    composition: str = Field(description='nominal material composition with basis marker, e.g., "AlCoCrFeNi2.5@at", "AlCoCrFeNi2.1@wt", "AlCoCrFeNi@at+Al2O3@wt[5%]"')
    label: Optional[str] = Field(description='label for the sample, normally abbreviation like A1, B2, etc.')

class PhaseRecord(BaseModel):
    metadata: MetaData
    phase: Phase

class StrengthRecord(BaseModel):
    metadata: MetaData
    strength: Strength

class GrainSizeRecord(BaseModel):
    metadata: MetaData
    grain_size: GrainSize

class PhaseRecords(BaseModel):
    records: Optional[List[PhaseRecord]]

class StrengthRecords(BaseModel):
    records: Optional[List[StrengthRecord]]

class GrainSizeRecords(BaseModel):
    records: Optional[List[GrainSizeRecord]]

class SynthesisRecord(BaseModel):
    metadata: MetaData
    synthesis: Synthesis

class SynthesisRecords(BaseModel):
    """If there are multiple samples fabricated in synthesis section, return multiple records."""
    records: Optional[List[SynthesisRecord]]