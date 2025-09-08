CATEGORIZE_DSPY = """You are analyzing the experimental section of a high entropy alloys (HEAs) research paper. Your task is to determine whether the author has distinctly identified each fabricated material.

**Task**: Return True if ALL materials are distinctly identified, False otherwise.

**Material Identification Requirements**:
- **Chemical composition**: Specific elemental ratios (e.g., "Mn0.2CoCrNi", "Al0.3CoCrFeNi0.5")
  - General formulas like "MnxCoCrNi" are NOT acceptable
- **Explicit symbol**: Clear identifier like "A0", "HEA-1", "Sample-700C"
  - Descriptive phrases without explicit symbols are NOT acceptable

**Evaluation Rules**:

**Case 1 - Single processing route, single sample**:
- True: If chemical composition is provided
- False: If composition is missing

**Case 2 - Single processing route, multiple samples with different compositions**:
- True: If every sample has either specific composition OR explicit symbol
- False: If any sample lacks both composition and symbol

**Case 3 - Multiple processing routes, same composition**:
(Different methods, parameters, temperatures, durations, etc.)
- True: If every sample is explicitly labeled with unique identifiers
- False: If any sample lacks clear labeling

**Case 4 - Multiple processing routes, different compositions**:
- True: If every sample has either specific composition OR explicit symbol
- False: If any sample lacks both composition and symbol"""

GROUP_DESCRIPTIONS = """You are given a list of material descriptions extracted from a research paper on high entropy alloys (HEAs). Each description may include information about composition, processing steps, and other identifiers. The descriptions may refer to the same material using slightly different wording, especially regarding processing conditions (e.g., "annealed", "after welding", "HPT-processed", etc.).

Here is the synthesis paragraph for additional context:
{context}

Your task is to:

Group the descriptions so that each group represents a distinct material (i.e., same composition and processing history), even if the wording varies. If the processing parameters such as temperature, duration, number of cycles/times, or other specific processing conditions are different, you must separate them into different groups, even if the composition is the same.
For each group, provide a representative and concise processing description that best summarizes the material and its processing route. 
If a material does not have any processing description, just use the composition as its representative description.
Output the result as a JSON array, where each element is an object with the following fields:

"group_id": an integer starting from 1
"representative_processing_description": a string summarizing the group
"descriptions": a list of the original descriptions belonging to this group
Example output: [ {{ "group_id": 1, "representative_processing_description": "FeCoNiCrMn alloy annealed at 1000°C", "descriptions": [ "FeCoNiCrMn alloy annealed at 1000°C for 2h", "FeCoNiCrMn, after annealing at 1000°C" ] }}, {{ "group_id": 2, "representative_processing_description": "FeCoNiCrMn alloy processed by high-pressure torsion (HPT)", "descriptions": [ "FeCoNiCrMn alloy, HPT-processed", "FeCoNiCrMn, after high-pressure torsion" ] }}, {{ "group_id": 3, "representative_processing_description": "V10Cr15Mn5Fe35Co10Ni25", "descriptions": [ "V10Cr15Mn5Fe35Co10Ni25" ] }} ]

Now, given the following list of material descriptions, perform the grouping and provide representative processing descriptions for each group in the specified JSON format:
{descriptions}
start with ```json, end with ```
"""
