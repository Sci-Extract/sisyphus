from router import router_api
router_api()
import warnings
import logging
from typing import Callable, Literal
from functools import partial

import dspy
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from sisyphus.chain import Filter, Writer
from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb, get_title_abs, render_docs
from sisyphus.urgent.json_schemas import StrengthRecords, PhaseRecords, GrainSizeRecords
import sisyphus.urgent.json_schemas_no_syn
from sisyphus.chain import Paragraph, ParagraphExtend
from sisyphus.strategy.utils import get_paras_with_props, get_synthesis_paras
from sisyphus.urgent.properties_extraction import extract_func_wrapper
from sisyphus.urgent.entity_resolution import entity_resolution_llms, entity_resolution_rule
from sisyphus.urgent.merge import merge, REFERRED

from prompt import simple_prompt_template, simple_prompt_template_no_syn, phase_instruction, strength_instruction, grain_size_instruction
    

warnings.filterwarnings('ignore', category=RuntimeWarning, module='pydantic') # the case that we convert json string to python object trigger pydantic warning

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('more_than_20.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

model = ChatOpenAI(temperature=0, model='gpt-4.1')
lm = dspy.LM('openai/gpt-4.1-mini')

class ClassifyPaper(dspy.Signature):
    """assign label to HEAs (high entropy alloys) paper based on their title and abstract."""
    context: str = dspy.InputField(desc='Title and abstract of the paper')
    label: Literal['hea_experimental', 'hea_theoretical', 'irrelevant'] = dspy.OutputField(desc="Pay attention to keywords such as 'molecular dynamics' or 'machine learning,' which should be labeled as hea_theoretical. Label keywords related to fabrication processes as hea_experimental.")
    mechanical_relevancy: bool = dspy.OutputField(desc='whether this paper describe the mechanical properties such as tensile or compressive')
classifier_paper = dspy.ChainOfThought(signature=ClassifyPaper)

def paper_filter(docs):
    title, abstract = get_title_abs(docs)
    prediction = classifier_paper(context=render_docs(abstract, title))
    if prediction.label == 'hea_experimental' and prediction.mechanical_relevancy:
        return docs
    return

def _make_extractor(prompt_template, output_model, property_labels, context_labels, instruction, type_name, has_synthesis=True):
    """Helper to create an extractor partial to avoid repeated boilerplate."""
    # Wrap the newer `extract_property_` which composes prompt + model
    return partial(
        extract_property_,
        property_labels=property_labels,
        context_labels=context_labels,
        has_synthesis=has_synthesis,
        prompt_template=prompt_template,
        instruction=instruction,
        chat_model=model,
        output_model=output_model,
        type=type_name,
    )

def extract_property_(
        paragraphs: list[Paragraph],
        property_labels: list[str],
        context_labels: list[str],
        has_synthesis: bool,
        prompt_template: ChatPromptTemplate,
        instruction: str,
        chat_model: ChatOpenAI,
        output_model: BaseModel,
        **kwargs 
) -> ParagraphExtend:
    chain = prompt_template | chat_model.with_structured_output(output_model, method='json_schema')

    target_paras = get_paras_with_props(paragraphs, *property_labels, *context_labels)
    is_existence = get_paras_with_props(paragraphs, *property_labels)
    if not is_existence:
        return []

    if has_synthesis:
        paragraph = ParagraphExtend.from_paragraphs(target_paras, **kwargs)
        syn_para = ParagraphExtend.from_paragraphs(get_synthesis_paras(paragraphs)) 
        res = chain.invoke(
            {
                'property_instruction': instruction,
                'synthesis_para': syn_para.page_content,
                'property': paragraph.page_content
            }
        )
    else:
        paragraph = ParagraphExtend.from_paragraphs(target_paras, **kwargs)
        res = chain.invoke(
            {
                'property_instruction': instruction,
                'property': paragraph.page_content
            }
        )

    paragraph.set_data(res.records)
    if paragraph.data:
        return [paragraph]
    return []

# Extractors when synthesis paragraphs exist
extract_phase = _make_extractor(
    simple_prompt_template, PhaseRecords,
    ['phase'], ['composition'], phase_instruction, 'phase_extraction'
)
extract_strength = _make_extractor(
    simple_prompt_template, StrengthRecords,
    ['strength'], ['composition', 'strain_rate'], strength_instruction, 'strength_extraction'
)
extract_grainsize = _make_extractor(
    simple_prompt_template, GrainSizeRecords,
    ['grain_size'], ['composition'], grain_size_instruction, 'grain_size_extraction'
)

# Extractors when synthesis paragraphs do NOT exist
extract_phase_no_syn = _make_extractor(
    simple_prompt_template_no_syn, sisyphus.urgent.json_schemas_no_syn.PhaseRecords,
    ['phase'], ['composition'], phase_instruction, 'phase_extraction_no_syn', has_synthesis=False
)
extract_strength_no_syn = _make_extractor(
    simple_prompt_template_no_syn, sisyphus.urgent.json_schemas_no_syn.StrengthRecords,
    ['strength'], ['composition', 'strain_rate'], strength_instruction, 'strength_extraction_no_syn', has_synthesis=False
)
extract_grainsize_no_syn = _make_extractor(
    simple_prompt_template_no_syn, sisyphus.urgent.json_schemas_no_syn.GrainSizeRecords,
    ['grain_size'], ['composition'], grain_size_instruction, 'grain_size_extraction_no_syn', has_synthesis=False
)


def extract(paragraphs: list[Paragraph]):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    merged = []
    if syn_paras:=get_synthesis_paras(paragraphs):
        extractors = (extract_phase, extract_strength, extract_grainsize)
        expected_names = ['extract_phase', 'extract_strength', 'extract_grainsize']
        name_to_result = {name: [] for name in expected_names}

        with ThreadPoolExecutor(max_workers=3) as ex:
            future_to_name = {ex.submit(fn, paragraphs): name for fn, name in zip(extractors, expected_names)}
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    name_to_result[name] = fut.result() or []
                except Exception:
                    # keep failure of one extractor from stopping others
                    name_to_result[name] = []

        paras = (name_to_result.get('extract_phase', []) +
                 name_to_result.get('extract_strength', []) +
                 name_to_result.get('extract_grainsize', []))

        records_groups = []
        records = []
        for para in paras:
            records_groups.append(para.data)
            records_groups = [group for group in records_groups if group]  # filter out empty groups
            records.extend(para.data)
            records = [record for record in records if record]  # filter out empty records
        if records:
            metadata_referred = [record.metadata.model_dump() for record in records if getattr(record, REFERRED)]
            metadata_all = [record.metadata.model_dump() for record in records]
            metadata_groups = [[record.metadata.model_dump() for record in group if not getattr(record, REFERRED)] for group in records_groups]

            if len(metadata_groups) > 1:
                # safety cutoff ~ 20
                if sum(len(group) for group in metadata_groups) > 20:
                    logger.info('Skipping for DOI: %s', paragraphs[0].metadata.get('doi'))
                    return paras
                syn_text = ParagraphExtend.from_paragraphs(syn_paras).page_content
                resolved_metadata_groups = entity_resolution_llms(metadata_groups, model, syn_text) + entity_resolution_rule(metadata_referred, ['composition', 'label'])
            else:  # fallback to rule-based if only one group
                resolved_metadata_groups = entity_resolution_rule(metadata_all, ['composition', 'label'])

            merged = merge(resolved_metadata_groups, records)

    else:
        extractors = (extract_phase_no_syn, extract_strength_no_syn, extract_grainsize_no_syn)
        expected_names = ['extract_phase_no_syn', 'extract_strength_no_syn', 'extract_grainsize_no_syn']
        name_to_result = {name: [] for name in expected_names}

        with ThreadPoolExecutor(max_workers=3) as ex:
            future_to_name = {ex.submit(fn, paragraphs): name for fn, name in zip(extractors, expected_names)}
            for fut in as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    name_to_result[name] = fut.result() or []
                except Exception:
                    name_to_result[name] = []

        paras = (name_to_result.get('extract_phase_no_syn', []) +
                 name_to_result.get('extract_strength_no_syn', []) +
                 name_to_result.get('extract_grainsize_no_syn', []))

        records = []
        for para in paras:
            records.extend(para.data)
            records = [record for record in records if record]  # filter out empty records
        if records:
            metadata_all = [record.metadata.model_dump() for record in records]
            resolved_metadata_groups = entity_resolution_rule(metadata_all, ['composition', 'label'])

            merged = merge(resolved_metadata_groups, records)

    if merged:
        with open('merged_records_debug.jsonl', 'a') as f:
            import json
            to_write = {
                'DOI': paragraphs[0].metadata.get('doi'),
                'records': merged
            }
            json_str = json.dumps(to_write, ensure_ascii=False)
            f.write(json_str + "\n")
    return paras

labeled_database = get_plain_articledb('heas_labeled_extend')
labeled_getter = Filter(labeled_database)
result_db = get_create_resultdb('urgent_test')
writer = Writer(result_db)
def load(docs):
    return [Paragraph.from_labeled_document(doc, id_) for id_, doc in enumerate(docs)]
extract_chain = labeled_getter + load + extract + writer

from script.file_name_doi_conversion import doi_to_file_name
dois = [
    '10.1002/adem.201900587',
    '10.1002/adem.202200523',
    '10.1002/mawe.202300263',
    '10.1016/j.actamat.2021.117571',
    '10.1016/j.actamat.2024.120498'
]
file_names = [doi_to_file_name(doi) for doi in dois]
# extract_chain.compose(file_names[0])10.1002/mawe.202300263
extract_chain.compose(file_names[1])
from sisyphus.chain.chain_elements import run_chains_with_extarction_history_multi_threads
run_chains_with_extarction_history_multi_threads(extract_chain, 'heas_test', 5, 'urgent_test')