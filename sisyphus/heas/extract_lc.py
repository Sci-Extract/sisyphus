import re
import json
import logging
import warnings
from ast import literal_eval
from typing import Optional, Literal, List

import dspy
from pydantic import BaseModel, Field, create_model, field_validator, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import LengthFinishReasonError

from sisyphus.chain.chain_elements import DocInfo
from sisyphus.chain.constants import FAILED
from sisyphus.utils.helper_functions import render_docs, reorder_paras, render_docs_without_title, get_title_abs
from sisyphus.chain.paragraph import Paragraph, ParagraphExtend
from sisyphus.heas.synthesis import get_synthesis_prompt, get_synthesis_prompt_all
from sisyphus.heas.prompt import (
    SYSTEM_MESSAGE_SYN,
    SYSTEM_MESSAGE_NO_SYN,
    INSTRUCTION_TEMPLATE,
    strength_instruction,
    phase_instruction,
    grain_size_instruction,
)
from sisyphus.heas.models import MetaData, Strength, Phase, GrainSize, Synthesis


warnings.filterwarnings('ignore', category=UserWarning, module='pydantic') # avoide the case that we convert json string to python object will trigger pydantic warning

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('extract_lc.log')
fh.setLevel(logging.DEBUG)
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(fh)

def create_instruction_dynamic(properties: List[Literal['strength', 'phase', 'grain_size']], synthesis_instruction: str):
    instruction = INSTRUCTION_TEMPLATE
    if 'strength' in properties:
        instruction += "\n### **Strength**\n" + strength_instruction + '\n'
    if 'phase' in properties:
        instruction += "\n### **Phase**\n" + phase_instruction + '\n'
    if 'grain_size' in properties:
        instruction += "\n### **Grain size**\n" + grain_size_instruction + '\n'
    if synthesis_instruction:
        instruction += "\n### **Processes formatted**\n" + synthesis_instruction
    return instruction

template_with_syn = ChatPromptTemplate(
    [
        ('system', SYSTEM_MESSAGE_SYN),
        ('user', '[START OF PAPER]\n{paper}\n[END OF PAPER]\n\nInstruction:\n{instruction}')
    ]
)

template_without_syn = ChatPromptTemplate(
    [
        ('system', SYSTEM_MESSAGE_NO_SYN),
        ('user', '[START OF PAPER]\n{paper}\n[END OF PAPER]\n\nInstruction:\n{instruction}')
    ]
)


# ======MODELS======
class Record(BaseModel):
    metadata: MetaData
    strength: List[Strength] = Field(description='tensile or compressive strength data')
    phase: List[Phase] = Field(description='phase information')
    grain_size: GrainSize = Field(description='grain size information')

class Records(BaseModel):
    records: List[Record]

def create_result_model_dynamic(properties: List[Literal['strength', 'phase', 'grain_size']], has_synthesis: bool):
    """Dynamically create a Pydantic model for Records based on the requested properties using create_model."""
    fields = {
        'metadata': (MetaData, ...)
    }

    if 'strength' in properties:
        fields['strength'] = (List[Strength], Field(..., description='tensile or compressive strength data'))
    if 'phase' in properties:
        fields['phase'] = (List[Phase], Field(..., description='phase information'))
    if 'grain_size' in properties:
        fields['grain_size'] = (List[GrainSize], Field(..., description='grain size information'))
    if has_synthesis:
        fields['synthesis'] = (Synthesis, Field(..., description='synthesis information'))

    Record = create_model('Record', __base__=BaseModel, **fields)
    # ensure dynamic models forbid extra properties so the LM JSON schema includes additionalProperties=false
    Record.model_config = ConfigDict(extra="forbid")
    doc = "List of extracted records, encourage to split into multiple records if processing parameters varied"
    Records = create_model('Records', __base__=BaseModel, records=(List[Record], Field(..., description='list of extracted records')), __doc__=doc)
    Records.model_config = ConfigDict(extra="forbid")
    return Records


# ======Extract======
def extract(paragraphs: list[Paragraph], extraction_model, synthesis_extract_model=dspy.LM('openai/gpt-4.1')):
    syn_paras = [para for para in paragraphs if para.is_synthesis]
    abstract_paras = [para for para in paragraphs if para.is_abstract()]
    intro_candidates = [
        p for p in paragraphs
        if isinstance(p.metadata, dict) and re.search(r'introduction', str(p.metadata.get('sub_titles', '')), re.I)
    ]
    last_intro_para = intro_candidates[-1:] if intro_candidates else []

    def paras_with_property(prop: str):
        """Return paragraphs that report a given property (safe against missing methods)."""
        return [p for p in paragraphs if hasattr(p, "has_property") and p.has_property(prop)]

    strength_paras = paras_with_property('strength')
    phase_paras = paras_with_property('phase')
    grain_size_paras = paras_with_property('grain_size')
    strain_rate_paras = paras_with_property('strain_rate')
    composition_paras = paras_with_property('composition')
    processing_param_paras = paras_with_property('processing_parameters')
    
    properties = []
    if strength_paras:
        properties.append('strength')
    if phase_paras:
        properties.append('phase')
    if grain_size_paras:
        properties.append('grain_size')
    if syn_paras:
        has_synthesis = True
    else:
        has_synthesis = False

    if len(properties) == 0:
        # try abstract
        if abstract_paras:
            chain = template_without_syn | extraction_model.with_structured_output(Records, method='json_schema')
            para_extend = ParagraphExtend.from_paragraphs(abstract_paras)
            records = chain.invoke({'paper': para_extend.page_content, 'instruction': INSTRUCTION_TEMPLATE}).records
            para_extend.set_data(records)
            return para_extend
        return

    if has_synthesis:
        result_model = create_result_model_dynamic(properties, True)
        synthesis_prompt = get_synthesis_prompt(render_docs_without_title(syn_paras), lm=synthesis_extract_model)
        instruction = create_instruction_dynamic(properties, synthesis_prompt)
        template = template_with_syn
    else:
        result_model = create_result_model_dynamic(properties, False)
        instruction = create_instruction_dynamic(properties, "")
        template = template_without_syn

    combined_paras = reorder_paras(abstract_paras + syn_paras + phase_paras + composition_paras + strength_paras + last_intro_para + grain_size_paras + strain_rate_paras + processing_param_paras)
    para_extend = ParagraphExtend.from_paragraphs(combined_paras)
    # with open("debug_extract_lc_prompt.txt", "w+", encoding="utf-8") as f:
    #     f.write(f'paper:{para_extend.page_content}\n\n instruction:{instruction}')
    # return
    
    chain = template | extraction_model.with_structured_output(result_model, method='json_schema')
    try:
        # debug
        # with open("debug_extract_lc_prompt.txt", "w+", encoding="utf-8") as f:
        #     f.write(instruction)
        records = chain.invoke({'paper': para_extend.page_content, 'instruction': instruction}).records
        para_extend.set_data(records)
    except LengthFinishReasonError:
        source = paragraphs[0].metadata['source']
        logger.exception('For source: %s', source, exc_info=1)
        return FAILED
    return para_extend

def extract_bulk(paragraphs, extraction_model):
    template = template_with_syn
    para_extend = ParagraphExtend.from_paragraphs(paragraphs)
    synthesis_instr = get_synthesis_prompt_all()
    def construct_instruction(synthesis_instruction):
        instruction = INSTRUCTION_TEMPLATE
        instruction += "\n### **Strength**\n" + strength_instruction + '\n'
        instruction += "\n### **Phase**\n" + phase_instruction + '\n'
        instruction += "\n### **Grain size**\n" + grain_size_instruction + '\n'
        instruction += "\n### **Processes formatted**\n" + synthesis_instruction
        return instruction
    result_model = create_result_model_dynamic(['strength', 'phase', 'grain_size'], has_synthesis=True)
    chain = template | extraction_model.with_structured_output(result_model, method='json_schema')
    try:
        records = chain.invoke({'paper': para_extend.page_content, 'instruction': construct_instruction(synthesis_instr)}).records
        para_extend.set_data(records)
    except LengthFinishReasonError:
        source = paragraphs[0].metadata['source']
        logger.exception('For source: %s', source, exc_info=1)
        return FAILED
    return para_extend
