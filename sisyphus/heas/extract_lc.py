import re
import logging
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import LengthFinishReasonError

from sisyphus.chain.chain_elements import DocInfo
from sisyphus.chain.constants import FAILED
from sisyphus.utils.helper_functions import render_docs, reorder_paras, render_docs_without_title, get_title_abs
from .paragraph import Paragraph, ParagraphExtend
from .synthesis import get_synthesis_prompt


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('extract_lc.log')
fh.setLevel(logging.DEBUG)
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(fh)

SYSTEM_MESSAGE = """You are an expert at structured data extraction from HEAs (high entropy alloys) domain. You will be given unstructured text from a research paper and should convert it into the given structure"""
INSTRUCTION_TEMPLATE = \
"""Extract all HEAs' tensile and compressive properties along with their synthesis methods from the text.  
- **Composition Format:** (e.g., `Hf0.5Mo0.5NbTiZrC0.3`).  
- **Handling Unknown Compositions:** If a nominal composition is missing (e.g., due to doping), use a **descriptive name** (e.g., `W-Co0.5Cr0.3FeMnNi`).  
- **Acronyms Prohibited:** Do **not** use labels like `HEA-1` or `Sample A`.  
---
### **General Guidelines:**  
- **Extract data for all materials** with reported mechanical properties, even if they are:  
  - Mentioned in **comparisons** with other samples.  
  - Referenced from a **previous study** but include numerical properties in the current text.  
  - Not the main focus of the paragraph.
- If multiple materials are described under different conditions, **each must be recorded separately** with its corresponding processing conditions.  

### **Mechanical Properties to Extract:**  
- **Tensile properties:** yield strength, ultimate tensile strength, elongation
- **Compressive properties:** Compressive yield strength, ultimate compressive strength, compressive strain
- **Synthesis method** If present, extract the synthesis information from text

### **Properties to Explicitly EXCLUDE:**  
- **Shear strength/stress** (do NOT misinterpret this as yield strength)
- **Critical resolved shear stress (CRSS)**
- **Fracture strength**
- **Hardness** (e.g., Vickers, Brinell, Rockwell)
- **Fatigue strength**
- **Young's modulus**

### **Extraction Rules:**  
- If properties are reported as "mean ± standard deviation," use the **mean value** (e.g., **700 ± 30 → 700**).  
- If properties are reported as a range, use the **lower bound** (e.g., **500–600 → 500**).  
- **Prioritize table values** over text if there is a conflict. 
- **If a material is mentioned in comparison to another material, still extract its properties.**  
- **If a material from a previous study is mentioned with numerical values, extract it as well.**

### **Processes formatted**  
{synthesis_prompt}

### **Ensuring Comprehensive Extraction:**  
- **Materials with mechanical properties mentioned in comparisons must be extracted.**  
- **Materials referenced from past studies must be extracted if numerical values are given.**
- **Materials which are not the main focus of the paper should also be extracted if they have mechanical properties.**
"""

class StrengthRecord(BaseModel):
    composition: str = Field()
    composition_type: Literal['atomic', 'weight']
    phase: Optional[list[str]]
    ys: Optional[float] = Field(description='convert to MPa if the unit is not MPa, e.g. 1 GPa -> 1000 MPa')
    uts: Optional[float] = Field(description='convert to MPa if the unit is not MPa, e.g. 1 GPa -> 1000 MPa')
    strain: Optional[float]
    processes: Optional[list[str]] = Field(description='eg., [induction melting: atmosphere: Ar, remelting times: 5, annealed: temperature: 700 °C, duration: 1 h, homogenized: temperature: 1200 °C, duration: 2 h]')
    test_type: Literal['tensile', 'compressive']
    test_temperature: str = Field(description='if not explicitly mentioned, record as 25 °C')

class Records(BaseModel):
    records: Optional[list[StrengthRecord]]


model = ChatOpenAI(model='gpt-4o', temperature=0, max_tokens=15000)
template = ChatPromptTemplate(
    [
        ('system', SYSTEM_MESSAGE),
        ('user', '[START OF PAPER]\n{paper}\n[END OF PAPER]\n\nInstruction:\n{instruction}')
    ]
)

def extract(paragraphs: list[Paragraph]):
    syn_paras = [para for para in paragraphs if para.is_synthesis]
    strength_paras = [para for para in paragraphs if para.has_property('strength')]
    phase_paras = [para for para in paragraphs if para.has_property('phase')]
    composition_paras = [para for para in paragraphs if para.has_property('composition')]
    last_intro_para = [para for para in paragraphs if re.search(r'introduction', para.metadata['sub_titles'], re.I)][-1:]

    # get synthesis prompt
    synthesis_prompt = get_synthesis_prompt(render_docs_without_title(syn_paras))

    # construct instruction
    instruction = INSTRUCTION_TEMPLATE.format(synthesis_prompt=synthesis_prompt)

    combined_paras = reorder_paras(syn_paras + phase_paras + composition_paras + strength_paras + last_intro_para)
    title, _ = get_title_abs(paragraphs)
    para_extend = ParagraphExtend.merge_paras(combined_paras, metadata={'doi': paragraphs[0].metadata['doi'], 'source': paragraphs[0].metadata['source']}, title=title)
    
    chain = template | model.with_structured_output(Records, method='json_schema')
    try:
        records = chain.invoke({'paper': para_extend.page_content, 'instruction': instruction}).records
    except LengthFinishReasonError:
        source = paragraphs[0].metadata['source']
        logger.exception('For source: %s', source, exc_info=1)
        return FAILED
    if records:
        return [DocInfo(para_extend, records)]

