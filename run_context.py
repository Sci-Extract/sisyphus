from typing import Optional
from functools import partial

import dspy
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from sisyphus.heas.label import label_paras
from sisyphus.chain.paragraph import Paragraph, ParagraphExtend
from sisyphus.chain import Filter, Writer
from sisyphus.strategy.run_strategy import extract_main
from sisyphus.strategy.pydantic_models_general import Processing, Material, MaterialDescriptionBase
from sisyphus.strategy.utils import build_process_agent, build_property_agent, build_result_model_contextualized, get_paras_with_props, get_synthesis_paras
from sisyphus.heas.prompt import *
from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb
from sisyphus.heas.synthesis import get_synthesis_prompt


lm = dspy.LM('openai/gpt-4.1-mini')
dspy.configure(lm=lm)
chat_model = ChatOpenAI(model='gpt-4.1-mini')

class StrengthTestBase(BaseModel):
    """Tensile/Compressive test results"""
    ys: Optional[str] = Field(description="Yield strength with unit")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain with unit")
    temperature: Optional[str] = Field(description="Test temperature with unit")
    strain_rate: Optional[str] = Field(description="Strain rate with unit")
    other_test_conditions: str = Field(description="Other test conditions, like in salt, hydrogen charging, etc.")

class PhaseInfo(BaseModel):
    """Phase information"""
    phases: Optional[str] = Field(description="List of phases present in the material")

strength_agent = build_property_agent(
    EXTRACT_PROPERTY_SYS_GENERIC_PROMPT,
    STRENGTH_PROMPT,
    build_result_model_contextualized('Strength', 'Extract tensile/compressive test information from provided text', StrengthTestBase, MaterialDescriptionBase),
    chat_model
)

phase_agent = build_property_agent(
    EXTRACT_PROPERTY_SYS_GENERIC_PROMPT,
    PHASE_PROMPT,
    build_result_model_contextualized('Phase', 'Extract phase information', PhaseInfo, MaterialDescriptionBase),
    chat_model
)

process_agent = build_process_agent(
    EXTRACT_PROCESS_SYS_GENERIC_PROMPT,
    PROCESS_PROMPT,
    Processing,
    chat_model
)

agents_d = {
    'strength': strength_agent,
    'phase': phase_agent
}

def reconstruct_p(paragraphs):
    p_str = ParagraphExtend.from_paragraphs(get_paras_with_props(paragraphs, 'strength'), type='strength')
    p_phase = ParagraphExtend.from_paragraphs(get_paras_with_props(paragraphs, 'phase'), type='phase')
    syn_paras = get_synthesis_paras(paragraphs)
    syn_paras.extend(get_paras_with_props(paragraphs, 'composition'))
    p_exp = ParagraphExtend.from_paragraphs(syn_paras, type='synthesis')
    
    return {
        "strength": p_str,
        "phase": p_phase,
        "synthesis": p_exp
    }

extract = partial(extract_main, reconstruct_paragraph=reconstruct_p, property_agents_d=agents_d, formatted_func=get_synthesis_prompt, synthesis_agent=process_agent)

db = get_plain_articledb('heas_1531')
getter = Filter(db)

chain = getter + label_paras + extract
chain.compose('10.1002&sol;adem.201900587.html')