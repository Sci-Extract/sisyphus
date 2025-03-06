from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional, Literal

import dspy
from pydantic import BaseModel, Field


strength_prompt = \
"""Follow these specific guidelines:

Data Extraction Rules:
1. For statistical values (mean ± standard deviation), extract ONLY the mean value (e.g., 700 ± 30 MPa → 700 MPa)
2. For ranges, extract ONLY the lower bound value (e.g., 500–600 MPa → 500 MPa)
3. When conflicts exist between text and tables, ALWAYS prioritize values from tables
4. Include units with all numerical values (MPa, %, etc.)
5. Extract values from both original experimental tests AND referenced data from other papers

Processing Route Matching:
1. Match each material to the MOST SPECIFIC processing route that applies, but ONLY for materials directly tested in this paper
2. Capture materials at ALL stages of processing (including intermediate products) for directly tested materials
3. For materials that underwent only a subset of steps in a full route, match to the appropriate partial route
4. Clearly identify when a material represents an intermediate stage of processing

Required Information Per Record:
1. Material composition: which needs to be a valid formula in the domain of HEAs, abbreviations are not allowed
2. Only record YS, UTS, elongation (compression or tensile), do not extract property of shear stress (CRSS). If none of these properties are present, do not extract it
3. For directly tested materials: include corresponding processing route and steps ID from provided list
4. For referenced materials: do NOT assign any processing route and leave processing route field empty

Additional Requirements:
1. Extract ALL materials with numerical properties, including those referenced from previous studies
2. For directly tested materials, include reference to the specific processing route number for each entry
3. For referenced materials, extract the property values WITHOUT assigning any processing method
4. YOU MUST extract all referenced material properties mentioned in the text

"""

phase_prompt = \
"""Extract phase information from the text and match each material record to its corresponding processing route from the provided list. Follow these specific guidelines:

Processing Route Matching:
1. Match each material to the MOST SPECIFIC processing route that applies, but ONLY for materials directly tested in this paper
2. Capture materials at ALL stages of processing (including intermediate products) for directly tested materials
3. For materials that underwent only a subset of steps in a full route, match to the appropriate partial route
4. Clearly identify when a material represents an intermediate stage of processingrocessing

Required Information Per Record:
1. Material composition: which needs to be a valid formula in the domain of HEAs, abbreviations are not allowed

Phase Extraction Rules:
1. Common phases include FCC (face-centered cubic), BCC (body-centered cubic), HCP (hexagonal close-packed), B2 (ordered bcc), L12 (ordered fcc), Laves, sigma, Intermetallic, etc.
2. Only record phase information relevant to high entropy alloys.
3. There can be situations when samples which underwent different processes are the same phase, you should record them separately.

"""

hardness_prompt = \
"""Extract hardness information from the text and match each material record to its corresponding processing route from the provided list. Follow these specific guidelines:

Processing Route Matching:
1. Match each material to the MOST SPECIFIC processing route that applies, but ONLY for materials directly tested in this paper
2. Capture materials at ALL stages of processing (including intermediate products) for directly tested materials
3. For materials that underwent only a subset of steps in a full route, match to the appropriate partial route
4. Clearly identify when a material represents an intermediate stage of processing

Required Information Per Record:
1. Material composition: which needs to be a valid formula in the domain of HEAs, abbreviations are not allowed

"""

grain_size_prompt = \
"""Extract grain size information from the text and match each material record to its corresponding processing route from the provided list. Follow these specific guidelines:

Processing Route Matching:
1. Match each material to the MOST SPECIFIC processing route that applies, but ONLY for materials directly tested in this paper
2. Capture materials at ALL stages of processing (including intermediate products) for directly tested materials
3. For materials that underwent only a subset of steps in a full route, match to the appropriate partial route
4. Clearly identify when a material represents an intermediate stage of processing

Required Information Per Record:
1. Material composition: which needs to be a valid formula in the domain of HEAs, abbreviations are not allowed

Grain size Extraction Rules:
1. If you can't find the text explicitly stated grain size in the paper, return an empty list.

"""

youngs_modulus_prompt = \
"""Extract Young's modulus information from the text and match each material record to its corresponding processing route from the provided list. Follow these specific guidelines:

Processing Route Matching:
1. Match each material to the MOST SPECIFIC processing route that applies, but ONLY for materials directly tested in this paper
2. Capture materials at ALL stages of processing (including intermediate products) for directly tested materials
3. For materials that underwent only a subset of steps in a full route, match to the appropriate partial route
4. Clearly identify when a material represents an intermediate stage of processing

Required Information Per Record:
1. Material composition: which needs to be a valid formula in the domain of HEAs, abbreviations are not allowed

Youngs modulus Extraction Rules:
1. only extract youngs modulus (elastic modulus) information from the text, do not extract other modulus information (e.g. bulk modulus, shear modulus, etc.).

"""

label_properties_prompt = \
"""First, determine whether properties exist or not in the provided paragraph. All properties (except phase) must have a float value to be considered valid. If a property doesn't have a float value, do not extract it. If no valid properties exist in the paragraph, return an empty list.

If properties exist, extract all valid properties. Properties must be one of the following:
- strength: Only includes tensile or compressive stress-strain information (such as yield strength, ultimate tensile strength, or strain values). Must have both a numeric value and unit. Explicitly EXCLUDE shear stress and corrosion resistance properties.
- phase: Micro-structure of HEAs, including FCC, BCC, HCP, B2, L12, Laves, sigma, Intermetallic. Usually characterized by XRD or SEM methods. This is the only property that doesn't require a float value.
- hardness: Vickers hardness measurements only. Must have a numeric value along with it.
- grain_size: Only the paragraph explicitly stated grain size along with its value in the text. Do NOT include other parameters such as lattice diameter or lattice parameters.
- youngs_modulus: Young's modulus measurements, with units in GPa. Must include a numeric value, only considering youngs modulus, not other modulus values (e.g. bulk modulus, shear modulus, etc.).

Be very strict in your extraction: except for phase, all other properties require a numeric value with appropriate units to be considered valid.

"""

class LabelProperties(dspy.Signature):
    paragraph: str = dspy.InputField()
    properties: list[Literal['strength', 'phase', 'hardness', 'grain_size', 'youngs_modulus']] = dspy.OutputField(desc='Return an empty list if no properties exist in the paragraph.')


class PropertyMeta(BaseModel):
    route_id: int = Field(..., description='Material processing route id')
    step_ids: list[int] = Field(..., description='Material processing steps')

class Strength(BaseModel):
    meta: Optional[PropertyMeta]
    composition: str = Field(..., description='The nominal chemical composition of the alloy. Ensure the validity of the formula, e.g. Hf0.5Mo0.5NbTiZrC0.3.')
    composition_type: Literal['at', 'wt'] = Field(..., description='The type of composition, either at.% or wt.%')
    ys: Optional[float] = Field(..., description='the value of yield strength (for tensile tests) or compression yield strength (for compression tests)')
    ys_unit: str
    uts: Optional[float] = Field(..., description='the value of ultimate tensile strength (for tensile tests) or maximum compression strength (for compression tests)')
    uts_unit: str
    strain: Optional[float] = Field(..., description='the value of elongation at break (for tensile tests) or maximum compression strain (for compression tests)')
    strain_unit: str
    test_type: Literal['tensile', 'compressive']
    test_temperature: str = Field(description='The temperature at which the mechanical properties were tested, e.g. 25 °C. If the temperature is given in Kelvin, convert it to Celsius by subtracting 273. If not mentioned in the text, record it as 25 °C.')

class ExtractStrength(dspy.Signature):
    context: str = dspy.InputField(desc='context include HEAs composition and synthesis informations')
    routes: str = dspy.InputField(desc='processing routes extracted from synthesis section')
    paragraph: str = dspy.InputField()
    properties: list[Strength] = dspy.OutputField()

class Phase(BaseModel):
    meta: Optional[PropertyMeta]
    composition: str = Field(..., description='The nominal chemical composition of the alloy. Ensure the validity of the formula, e.g. Hf0.5Mo0.5NbTiZrC0.3.')
    composition_type: Literal['at', 'wt'] = Field(..., description='The type of composition, either at.% or wt.%')
    phases: list[str] = Field(..., description='List of phases of the alloy, main phase should be the first element in the list. Common phases include FCC, BCC, HCP, B2, L12, Laves, sigma, Intermetallic.')

class ExtractPhase(dspy.Signature):
    context: str = dspy.InputField(desc='context include HEAs composition and synthesis informations')
    routes: str = dspy.InputField(desc='processing routes, each route is unique, but note that material can processed with only partial of steps in the route')
    paragraph: str = dspy.InputField()
    properties: list[Phase] = dspy.OutputField()

class Hardness(BaseModel):
    meta: Optional[PropertyMeta]
    composition: str = Field(..., description='The nominal chemical composition of the alloy. Ensure the validity of the formula, e.g. Hf0.5Mo0.5NbTiZrC0.3.')
    composition_type: Literal['at', 'wt'] = Field(..., description='The type of composition, either at.% or wt.%')
    hardness: float = Field(..., description='Vickers hardness value')
    hardness_unit: str
    applied_load: Optional[float] = Field(..., description='Vickers Hardness test performed with a load if given')
    applied_load_unit: Optional[str] = Field(..., description='Unit of applied load, usually kgf')

class ExtractHardness(dspy.Signature):
    context: str = dspy.InputField(desc='context include HEAs composition and synthesis informations')
    routes: str = dspy.InputField(desc='processing routes, each route is unique, but note that material can processed with only partial of steps in the route')
    paragraph: str = dspy.InputField()
    properties: list[Hardness] = dspy.OutputField()

class GrainSize(BaseModel):
    meta: Optional[PropertyMeta]
    composition: str = Field(..., description='The nominal chemical composition of the alloy. Ensure the validity of the formula, e.g. Hf0.5Mo0.5NbTiZrC0.3.')
    composition_type: Literal['at', 'wt'] = Field(..., description='The type of composition, either at.% or wt.%')
    grain_size: float = Field(..., description='Grain size value')
    grain_size_unit: str

class ExtractGrainSize(dspy.Signature):
    context: str = dspy.InputField(desc='context include HEAs composition and synthesis informations')
    routes: str = dspy.InputField(desc='processing routes, each route is unique, but note that material can processed with only partial of steps in the route')
    paragraph: str = dspy.InputField()
    properties: list[GrainSize] = dspy.OutputField()

class YoungsModulus(BaseModel):
    meta: Optional[PropertyMeta]
    composition: str = Field(..., description='The nominal chemical composition of the alloy. Ensure the validity of the formula, e.g. Hf0.5Mo0.5NbTiZrC0.3.')
    composition_type: Literal['at', 'wt'] = Field(..., description='The type of composition, either at.% or wt.%')
    youngs_modulus: float = Field(..., description="Young's modulus value")
    youngs_modulus_unit: str

class ExtractYoungsModulus(dspy.Signature):
    context: str = dspy.InputField(desc='context include HEAs composition and synthesis informations')
    routes: str = dspy.InputField(desc='processing routes, each route is unique, but note that material can processed with only partial of steps in the route')
    paragraph: str = dspy.InputField()
    properties: list[YoungsModulus] = dspy.OutputField()


LabelProperties.__doc__ = label_properties_prompt
ExtractStrength.__doc__ = strength_prompt
ExtractPhase.__doc__ = phase_prompt
ExtractHardness.__doc__ = hardness_prompt
ExtractGrainSize.__doc__ = grain_size_prompt
ExtractYoungsModulus.__doc__ = youngs_modulus_prompt

label_agent = dspy.ChainOfThought(LabelProperties)
property_extract_agents = {
    "strength": dspy.Predict(ExtractStrength),
    "phase": dspy.Predict(ExtractPhase),
    "hardness": dspy.Predict(ExtractHardness),
    "grain_size": dspy.Predict(ExtractGrainSize),
    "youngs_modulus": dspy.Predict(ExtractYoungsModulus)
}

def label_properties(paragraphs):
    """label properties for paragraphs"""
    with ThreadPoolExecutor(5) as executor:
        futures = [executor.submit(label_agent, paragraph=paragraph.page_content) for paragraph in paragraphs]
        future_para = dict(zip(futures, paragraphs))
        for future in as_completed(futures):
            properties = future.result().properties
            properties_filtered = [prop for prop in properties if prop in property_extract_agents]
            future_para[future].set_types(properties_filtered)
    return paragraphs
