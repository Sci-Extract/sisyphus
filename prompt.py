from langchain.prompts import ChatPromptTemplate

simple_prompt_template = ChatPromptTemplate.from_messages([
    ('user',"""
You are tasked with extracting material information from the provided text and outputting the data in a structured format. The format should be a list of dictionaries, where each dictionary contains the following metadata and property information. Specifically, the metadata has structure as follows:
metadata: {{
    "composition": "%s",
    "label": "%s",
    "procecessing_kw": "%s"
    }}
e.g., metadata: {{
    "composition": "Mn0.2CoCrNi@at",
    "label": "A1",
    "procecessing_kw": ["annealed at 800C for 1hr", "cold rolled 50%"]
    }}
Specifically for composition:
### **Composition Format:**.  
    - Use `@at` to denote atomic percent (at%) and `@wt` for weight percent (wt%). For simple alloys, keep original nominal composition + basis marker.
        - Example: `AlCoCrFeNi2.5@at`, `AlCoCrFeNi2.1@wt`
    - For composites, e.g., 1 wt% AlN nanoparticles added to AlCoCrFeNi (at. %)
        - composition: `AlCoCrFeNi@at+AlN@wt[1%]`

Notes:
The metadata field is an identifier of sample, which will be used in entity resolution later to merge different property records from the same sample, so property-relevant parameters should **NOT** be included in metadata.
The processing_kw field should only include keywords from the synthesis section and should be succinct (e.g., "annealed at 800°C", "cold rolled 50%").
Do not include processing steps related to property testing from property section in the processing_kw.
If no properties are found in the property section, return an empty list.

For property specific instruction: 
{property_instruction}

Synthesis section:
{synthesis_para}

Property section:
{property}
""")
]
)

simple_prompt_template_no_syn = ChatPromptTemplate.from_messages([
    ('user',"""
You are required to extract material information from text provided below and ouput desired format which generally a list of dictionaries, and for each includes metadata and property information. Return empty list if no property found. Specifically, the metadata has structure as follows:
metadata: {{
    "composition": "%s",
    "label": "%s",
    }}
e.g., metadata: {{
    "composition": "Mn0.2CoCrNi@at",
    "label": "A1",
    }}
Specifically for composition:
### **Composition Format:**.  
    - Use `@at` to denote atomic percent (at%) and `@wt` for weight percent (wt%). For simple alloys, keep original nominal composition + basis marker.
        - Example: `AlCoCrFeNi2.5@at`, `AlCoCrFeNi2.1@wt`
    - For composites, e.g., 1 wt% AlN nanoparticles added to AlCoCrFeNi (at. %)
        - composition: `AlCoCrFeNi@at+AlN@wt[1%]`

For property specific instruction: 
{property_instruction}

Property section:
{property}
""")
]
)
phase_instruction = """General:
1. Separate samples by parameters: Treat materials processed with different conditions (e.g., temperature, duration, method) as distinct samples.
2. Extract only explicitly stated data: Record only information directly stated in the text. Do not infer or assume.
3. Extract phase information per sample: Record phase data only if explicitly stated for each sample, if not available, exclude that sample from extraction.

Here are some common phase types:
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
strength_instruction = """Extract all mechanical property relevant to ys, uts and strain from the text

Follow these rules:
- Material composition should be in the form of nominal chemical formula in atom percentage, e.g., "Mn0.2CoCrNi", not any descriptive phrases.
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
