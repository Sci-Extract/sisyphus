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

EXTRACT_PROPERTY_SYS_GENERIC_PROMPT = """You are given a passage from a scientific article.
Your task is to extract all distinct material instances and their associated experimental properties.

A material instance is defined by:
- Composition: the nominal atomic ratio or formula.
- Symbol / Alias: any shorthand, explicit label, or descriptive identifier.
- Processing keywords: important processing steps that define the sample (e.g., melting, annealing, cold rolling, quenching, HPT).

Rules:
- Extract only material instances for which experimental property values are provided in the text. 
- Always keep material identities granular. Distinguish every material instance, even if the author refers to them collectively or uses broad terms (e.g., "all samples", "these alloys").
- When multiple materials share the same property value but are described separately or synthesized differently, extract each as a separate instance.

Output as a list of extracted material instances, with the above fields clearly filled.
"""

STRENGTH_PROMPT = """Extract all mechanical property relevant to ys, uts and strain from the text

Follow these rules:
- Material composition should be in the form of nominal chemical formula in atom percentage, e.g., "Mn0.2CoCrNi", not any descriptive phrases.
- Prioritize table values over text if there is a conflict. 
- If the value provided is a range, for example, "from 200 MPa to 300 MPa", extract it as "200-300 MPa".
- If the value is given as "greater than" or "less than", for example, "greater than 400 MPa", extract it as ">400 MPa".
- If the value is given as "approximately" or "around", for example, "approximately 250 MPa", extract it as "≈250 MPa".
- Otherwise, extract the value as it is.

text
{text}
"""

PHASE_PROMPT = """Extract the phase information from the text. Material composition should be in the form of chemical formula in atom percentage, e.g., "Mn0.2CoCrNi", not any descriptive phrases.

Important: If materials are synthesized with different parameters (e.g., temperature, duration, processing method), each should be considered a distinct sample. Extract phase information for each sample separately.

Here are some common phase types:
FCC, BCC, HCP, B2, intermetallic compounds (e.g., TiNi, Ti₂Ni, γ' precipitates, silicides, aluminides, sigma (σ) phases), carbides (e.g., WC), oxides (e.g., SiO₂), amorphous phases.

Guideline for phase extraction:
- If the author mentions ordered/disordered, include it in the phase information.
- Same phase can be present multiple times, e.g., "FCC, FCC" extract as "FCC, FCC".
- Main phase should be listed first, followed by secondary phases, and so on.

text
{text}
"""

EXTRACT_PROCESS_SYS_GENERIC_PROMPT = """You are an expert in extracting processing routes for materials from scientific texts.
"""

PROCESS_PROMPT = """Extract the processing route and nominal chemical composition for the specified material from the experimental section.
Guidance:
- Note that the given sample probably be one of many samples synthesised in the experimental section. You only need to extract the processing route for the specified sample.
- The composition of the material should be in the form of nominal formula in atom percentage, e.g., "Mn0.2CoCrNi", not any descriptive phrases.

Extract processing information using the predefined templates. Only include processing methods relevant to the material. For each included method, output all template fields - use empty strings for missing values, never omit fields. Use exact field names and maintain proper JSON format. 
{process_format}

Experimental section
{text}

The sample to extract processing route is:
**{material_description}**
"""

PROCESS_ISOLATED_PROMPT = """Extract the processing route and nominal chemical compositions from the text.
Guidance:
- If different materials synthesized, extract each material separately.
- If materials with different processing routes are described, extract each route separately.

Extract processing information using the predefined templates. Only include processing methods relevant to the material. For each included method, output all template fields - use empty strings for missing values, never omit fields. Use exact field names and maintain proper JSON format.
{process_format}

Experimental section
{text}
"""


# ======PROMPT======
SYSTEM_MESSAGE_SYN = \
"""You are an information extraction assistant for materials science. Your task is to read a full paper and output a list of material records. Each record represents a unique material entity, defined by both its composition and processing history.
For each entity, gather all explicitly reported properties such as mechanical properties, phase information, grain size, etc. When the material is tested under different conditions (e.g., varying testing temperatures, strain rates or hydrogen charging), results should be included in the same record as long as the material’s composition and processing history remain unchanged.
However, if there are variations in processing parameters (e.g., heat treatment temperature, annealing time, fabrication method), the resulting material properties should be recorded in separate records, as these changes constitute different processing histories.
Only create a record when at least one property is explicitly reported. Statements such as “all samples exhibit a certain property” count as explicit reporting for each corresponding entity. However, if it describes variations in processing but provides no property data for some of those variants, you should not create additional records for them. If processing history for a sample is not provided (e.g., only composition and properties are given, which is used for comparison), list it as a single record without processing details.
The goal is to precisely capture every distinct material entity and its associated properties, separating records where the processing history truly differs (including heat treatment of samples such as annealing, homogenization, aging etc.), but keeping differences in testing conditions (varied testing parameters, such as temperature) grouped within the same record when the material itself remains the same."""

SYSTEM_MESSAGE_NO_SYN = \
"""You are an information extraction assistant for materials science. Your task is to read a full review paper and output a list of material records. Each record represents one material entity defined primarily by its composition, because detailed synthesis or processing routes may not be provided in reviews.
For each entity, gather all explicitly reported properties (mechanical properties, phase information, grain size, etc.) wherever they appear in the text, figures, or captions. If multiple property values are reported for the same composition under different testing conditions, include all of them within the same record.
Do not create multiple records for the same composition unless the review clearly indicates that distinct samples with different processing histories were compared. Only create a record when at least one property is explicitly reported; do not produce entries based solely on composition names without associated data.
Your goal is to capture all material entities with reported property information while avoiding both over-merging and over-splitting."""

INSTRUCTION_TEMPLATE = \
"""Extract structured data from the provided text according to the following guidelines.  

### **Composition Format:**.  
    - Use `@at` to denote atomic percent (at%) and `@wt` for weight percent (wt%). For simple alloys, keep original nominal composition + basis marker.
        - Example: `AlCoCrFeNi2.5@at`, `AlCoCrFeNi2.1@wt`, `(FeCoNi)86Al7Ti7@at` (keep the parentheses)
    - For composites, e.g., 1 wt% AlN nanoparticles added to AlCoCrFeNi (at. %)
        - composition: `AlCoCrFeNi@at+AlN@wt[1%]`
    - **Acronyms Prohibited:** Do **not** use or includes labels like `HEA-1` or `Sample A`.  

### **General Guidelines:**  
- **Extract data for all materials** with required properties, even if they are:  
  - Mentioned in **comparisons** with other samples.  
  - Referenced from a **previous study** but include required properties in the current text.  
  - Not the main focus of the paragraph.
- **Synthesis extraction**:
  - Use only the **experimental/synthesis section** as the source of information.
  - Create split records when a processing parameter is reported as a range (e.g., 300–330 K) and later sections provide properties tied to specific values within that range (e.g., 300 K has prop1, 330 K has prop2). If the later property values are given only for the entire range and not for individual points, do not split the record and retain the range as a single parameter.
"""

phase_instruction = """Extract phase information from the text
hase types:
FCC, BCC, HCP, B2, intermetallic compounds (e.g., TiNi, Ti₂Ni, γ' precipitates, silicides, aluminides, sigma (σ) phases), carbides (e.g., WC), oxides (e.g., SiO₂), amorphous phases.
The dendrites themselves are made OF a phase (could be FCC, BCC, etc.), but "dendritic" itself is not a phase.
Other Similar Terms to Exclude from Phase:
Dendritic, equiaxed, columnar (grain morphology)
Lamellar, eutectic (phase arrangement)
Fine-grained, coarse-grained (grain size)

Guideline for phase extraction:
- If the author mentions ordered/disordered, include it in the phase information.
- Same phase can be present multiple times, e.g., "FCC, FCC" extract as "FCC, FCC".
- Main phase should be listed first, followed by secondary phases, and so on."""

strength_instruction = """Extract mechanical property relevant to ys, uts and strain from the text
Follow these rules:
- If tested under different setting (temperature, strain rate, etc), extract separately
- Only collect tensile/compressive strength and strain data, exclude all other mechanical properties.
- Strain should refer to fracture strain only.
- Prioritize table values over text if there is a conflict. 
- If the value provided is a range, for example, "from 200 MPa to 300 MPa", extract it as "200-300 MPa".
- If the value is given as "greater than" or "less than", for example, "greater than 400 MPa", extract it as ">400 MPa".
- If the value is given as "approximately" or "around", for example, "approximately 250 MPa", extract it as "~250 MPa".
- Otherwise, extract the value as it is."""

grain_size_instruction = """Extract the grain size information with units from the provided text. Do not record any other type of size information.

Follow these rules:
- If the value provided is a range, extract with range e.g., "20-30 nm".
- If the value is given as "greater than" or "less than", for example, "greater than 40 nm", extract it as ">40 nm".
- If the value is given as "approximately" or "around", for example, "approximately 20 nm", extract it as "~20 nm".
- Otherwise, extract the value as it is.
"""