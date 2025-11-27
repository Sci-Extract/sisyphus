from router import router_api
router_api()
import warnings
import logging
from typing import Callable, Literal
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from sisyphus.chain import Filter, Writer
from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb, get_title_abs, render_docs, render_docs_without_title
from sisyphus.urgent.json_schemas_no_syn import StrengthRecords, PhaseRecords, GrainSizeRecords, SynthesisRecords
from sisyphus.chain import Paragraph, ParagraphExtend
from sisyphus.strategy.utils import get_paras_with_props, get_synthesis_paras
from sisyphus.urgent.properties_extraction import extract_func_wrapper
from sisyphus.urgent.entity_resolution import entity_resolution_llms, entity_resolution_rule
from sisyphus.urgent.merge import merge, REFERRED
from sisyphus.heas.prompt import (
    INSTRUCTION_TEMPLATE,
    phase_instruction,
    strength_instruction,
    grain_size_instruction
)
from sisyphus.heas.synthesis import get_synthesis_prompt


warnings.filterwarnings('ignore', category=RuntimeWarning, module='pydantic') # the case that we convert json string to python object trigger pydantic warning

model = ChatOpenAI(temperature=0, model='gpt-4.1')

simple_prompt_template_no_syn = ChatPromptTemplate.from_messages([
    ('user',"""
You are required to extract material information from text provided below and ouput desired format which generally a list of dictionaries, and for each includes metadata and property/synthesis information. Return empty list if no property/synthesis found. Specifically, the metadata has structure as follows:
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

For property/synthesis specific instruction: 
{property_instruction}

Property/Synthesis section:
{property}
""")
]
)

def extract_property_(
        paragraphs: list[Paragraph],
        property_labels: list[str],
        context_labels: list[str],
        instruction: str,
        chat_model: ChatOpenAI,
        output_model: BaseModel,
        **kwargs 
) -> ParagraphExtend:
    chain = simple_prompt_template_no_syn | chat_model.with_structured_output(output_model, method='json_schema')

    target_paras = get_paras_with_props(paragraphs, *property_labels, *context_labels)
    is_existence = get_paras_with_props(paragraphs, *property_labels)
    if not is_existence:
        return
    paragraph = ParagraphExtend.from_paragraphs(target_paras, **kwargs)
    res = chain.invoke(
        {
            'property_instruction': instruction,
            'property': paragraph.page_content
        }
    )

    paragraph.set_data(res.records)
    return paragraph

def extract_synthesis_(
        paragraphs: list[Paragraph],
        context_labels: list[str],
        lm: dspy.LM,
        chat_model: ChatOpenAI,
        output_model: BaseModel,
        **kwargs
):
    syn_paras = get_synthesis_paras(paragraphs)
    context_paras = get_paras_with_props(paragraphs, *context_labels)
    if not syn_paras:
        return
    synthesis_prompt = get_synthesis_prompt(render_docs_without_title(syn_paras), lm=lm)
    chain = simple_prompt_template_no_syn | chat_model.with_structured_output(output_model, method='json_schema')
    paragraph = ParagraphExtend.from_paragraphs(syn_paras + context_paras, **kwargs)
    res = chain.invoke(
        {
            'property_instruction': synthesis_prompt,
            'property': paragraph.page_content
        }
    )
    paragraph.set_data(res.records)
    return paragraph


def extract(paragraphs: list[Paragraph]):
    extract_strength = partial(
        extract_property_,
        property_labels=['strength'],
        context_labels=['composition', 'strain_rate'],
        instruction=strength_instruction,
        chat_model=model,
        output_model=StrengthRecords
    )
    extract_phase = partial(
        extract_property_,
        property_labels=['phase'],
        context_labels=['composition'],
        instruction=phase_instruction,
        chat_model=model,
        output_model=PhaseRecords
    )
    extract_grain_size = partial(
        extract_property_,
        property_labels=['grain_size'],
        context_labels=['composition'],
        instruction=grain_size_instruction,
        chat_model=model,
        output_model=GrainSizeRecords
    )
    extract_synthesis = partial(
        extract_synthesis_,
        context_labels=['composition', 'processing_parameters'],
        lm=dspy.LM('openai/gpt-4.1'),
        chat_model=model,
        output_model=SynthesisRecords
    )
    extractors = [
        extract_strength,
        extract_phase,
        extract_grain_size,
        extract_synthesis
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        result_paras = executor.map(lambda func: func(paragraphs), extractors)
    merged = []
    records = []
    for para in result_paras:
        records.extend(para.data)
        records = [record for record in records if record]  # filter out empty records
    if records:
        metadata_all = [record.metadata.model_dump() for record in records]
        resolved_metadata_groups = entity_resolution_rule(metadata_all, ['composition', 'label'])
        merged = merge(resolved_metadata_groups, records)

    if merged:
        with open('heas_paragraph_wised.json', 'a') as f:
            import json
            data = json.load(f) if f.read() else []
            to_write = [{
                'doi': paragraphs[0].metadata.get('doi'),
                'Record': record
            } for record in merged]
            data.extend(to_write)
            json.dump(data, f, indent=2, ensure_ascii=False)

    return result_paras

    