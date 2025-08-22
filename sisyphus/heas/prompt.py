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

EXTRACT_PROPERTY_SYS_GENERIC_PROMPT = """You are an expert in extracting structured material properties from scientific texts. Your task is to extract the following properties from the provided text, which should includes all relevant information, even if it is used for comparison purposes."""

STRENGTH_PROMPT = """Extract all mechanical property relevant to ys, uts and strain from the text

Follow these rules:
- Material composition should be in the form of nominal chemical formula in atom percentage, e.g., "Mn0.2CoCrNi", not any descriptive phrases.
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
Requirements for the description field:
**Description of the material state/condition. Extract ONLY the processing history or treatment condition that defines the material itself, such as 'as-cast', 'annealed at 900°C for 2h', 'cold-rolled to 50% reduction', or 'solution-treated and aged'.
INCLUDE: Processing steps, heat treatments, mechanical working, surface treatments that permanently alter the material.
EXCLUDE: Testing conditions (e.g., 'tested at 700°C', 'measured in salt water'), sample geometry (e.g., 'dog-bone shaped'), or measurement parameters that describe the experimental setup rather than the material's intrinsic state.
If only composition is provided without any processing description, return 'None'.**"""

PROCESS_PROMPT = """Extract the processing route and nominal chemical composition for the specified material from the experimental section.
Guidance:
- Note that the given sample probably be one of many samples synthesised in the experimental section. You only need to extract the processing route for the specified sample.
- The composition of the material should be in the form of nominal formula in atom percentage, e.g., "Mn0.2CoCrNi", not any descriptive phrases.

And follow below format rules, not all processing methods required, depending on the material:
{process_format}

Experimental section
{text}

The sample to extract processing route for is:
**{material_description}**
"""