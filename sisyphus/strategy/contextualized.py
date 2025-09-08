# Implementation of contexualized extraction: Frist extract properties separately then link it with correponds synthesis routes via description of that material.
import json
import threading
import re
import warnings
from typing import Optional, Callable, Union, List, Dict, Any
from random import shuffle

from pydantic import BaseModel
from retry import retry
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sisyphus.utils.helper_functions import run_concurrently
from sisyphus.chain.paragraph import Paragraph, ParagraphExtend
from sisyphus.strategy.prompt_general import GROUP_DESCRIPTIONS
from sisyphus.strategy.default_chat_models import model
from sisyphus.strategy.utils import get_synthesis_paras
from sisyphus.strategy.pydantic_models_general import MaterialDescriptionBase, Processing, SafeDumpProcessing, ProcessingWithSymbol


warnings.filterwarnings('ignore', category=UserWarning, module='pydantic') # the case that we convert json string to python object trigger pydantic warning

group_prompt = ChatPromptTemplate.from_messages(
    [
        ('human', GROUP_DESCRIPTIONS)
    ]
)
group_chain = group_prompt | model | StrOutputParser()

def parse_result_string(result_str: str):
    """
    Parse a result string that may be wrapped in triple backticks and/or single quotes,
    and contains a JSON array, into a Python object.
    """
    # Remove leading/trailing whitespace and single quotes
    s = result_str.strip().strip("'").strip('"')
    # Extract JSON part between ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: try to find the first [ ... ] block
        match = re.search(r"(\[.*\])", s, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            raise ValueError("No JSON array found in result string.")
    return json.loads(json_str)

def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into batches of given size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

@retry(ValueError, tries=2)
def group_descriptions(batch: List[str], context: str) -> List[Dict]:
    """
    Placeholder for your LLM grouping logic.
    Each group is a dict with 'representative_processing_description' and 'descriptions'.
    """
    response = group_chain.invoke({"descriptions": batch, "context": context})
    return parse_result_string(response)

def recursive_hierarchical_grouping(descriptions: List[str], context: str, batch_size: int = 12, depth: int = 1, max_depth: int = 3) -> List[Dict]:
    """
    Recursively group descriptions in batches, stopping when group size < batch_size or max_depth is reached.
    Keeps track of all raw descriptions.
    """
    shuffle(descriptions)  # Shuffle to avoid order bias
    if len(descriptions) < batch_size or depth > max_depth:
        # Base case: group all remaining descriptions together
        return group_descriptions(descriptions, context)

    # First pass: group in batches
    first_pass_groups = []
    args = [(batch, context) for batch in batch_items(descriptions, batch_size)]
    for batch_groups in run_concurrently(group_descriptions, args):
        first_pass_groups.extend(batch_groups)

    # Prepare for next pass: flatten to representatives, keep mapping to raw descriptions
    next_pass_inputs = []
    for group in first_pass_groups:
        next_pass_inputs.append({
            "representative": group["representative_processing_description"],
            "raw_descriptions": group["descriptions"]
        })

    # If the number of groups is less than batch_size, but recursion is allowed, regroup these groups
    if len(next_pass_inputs) < batch_size:
        if depth >= max_depth:
            # Stop recursion, return as is
            return [{
                "representative_processing_description": item["representative"],
                "descriptions": item["raw_descriptions"]
            } for item in next_pass_inputs]
        # Otherwise, regroup the representatives
        reps = [item["representative"] for item in next_pass_inputs]
        # Run grouping logic again on these representatives
        regrouped = group_descriptions(reps, context)
        merged_groups = []
        for g in regrouped:
            merged_raw = []
            for rep in g["descriptions"]:
                for item in next_pass_inputs:
                    if item["representative"] == rep:
                        merged_raw.extend(item["raw_descriptions"])
            merged_groups.append({
                "representative_processing_description": g["representative_processing_description"],
                "descriptions": list(dict.fromkeys(merged_raw))
            })
        return merged_groups

    # Otherwise, recursively group the representatives
    reps = [item["representative"] for item in next_pass_inputs]
    higher_groups = recursive_hierarchical_grouping(reps, context, batch_size, depth=depth+1, max_depth=max_depth)

    # For each higher-level group, merge all raw descriptions from lower-level groups
    merged_groups = []
    for g in higher_groups:
        merged_raw = []
        for rep in g["descriptions"]:
            for item in next_pass_inputs:
                if item["representative"] == rep:
                    merged_raw.extend(item["raw_descriptions"])
        merged_groups.append({
            "representative_processing_description": g["representative_processing_description"],
            "descriptions": list(dict.fromkeys(merged_raw))
        })
    return merged_groups


def create_descriptions_dict(material_descriptions: List[MaterialDescriptionBase]):
    """Create a dictionary mapping description strings to MaterialDescriptionBase objects."""
    desc_to_obj = {}
    no_catgo_moterials = []
    for m in material_descriptions:
        if getattr(m, 'referred', False):
            no_catgo_moterials.append(m)
            continue
        comp = getattr(m, 'composition', None)
        sym = getattr(m, 'symbol', None)
        desc = getattr(m, 'description', None)
        if not any([comp, sym, desc]):
            no_catgo_moterials.append(m)
            continue
        key_parts = []
        if comp:
            key_parts.append(str(comp))
        if sym:
            key_parts.append(str(sym))
        if desc:
            key_parts.append(str(desc))
        key = "|".join(key_parts)
        if key not in desc_to_obj:
            desc_to_obj[key] = []
        desc_to_obj[key].append(m)
    return desc_to_obj, no_catgo_moterials

def merge_without_process(material_descriptions: List[MaterialDescriptionBase]) -> List[Dict]:
    """Merge material descriptions without any processing into groups based on identical descriptions."""
    desc_to_obj, no_catgo_moterials = create_descriptions_dict(material_descriptions)
    description_for_grouping = list(desc_to_obj.keys())
    if not description_for_grouping:
        # all materials are no categorization ones
        return [{
            "representative": None,
            "properties": [no_categorization_material],
            "synthesis": None
        } for no_categorization_material in no_catgo_moterials]
    
    # group
    grouped = recursive_hierarchical_grouping(description_for_grouping, "")
    results = []
    for group in grouped:
        all_objs = []
        for key in group["descriptions"]:
            all_objs.extend(desc_to_obj[key])
        results.append({
            "representative": group['representative_processing_description'],
            "properties": all_objs,
            "synthesis": None
        })
    # append no categorization properties
    for m in no_catgo_moterials:
        results.append({
            "representative": None,
            "properties": [m],
            "synthesis": None
        })
    return results
    

def merge_with_process(
    material_descriptions: list[MaterialDescriptionBase],
    processing_format,
    experimental_section: str,
    agent
) -> list[dict]:
    """
    Group material descriptions using recursive_hierarchical_grouping, skip referred.
    For each group, extract process and return a list of dicts with representative, properties, and processing object.
    """
    # Prepare descriptions for grouping: use a string key for each material
    desc_to_obj, no_categorization_materials = create_descriptions_dict(material_descriptions)
    descriptions_for_grouping = list(desc_to_obj.keys())
    if not descriptions_for_grouping:
        # all materials are no categorization ones
        return [{
            "representative": None,
            "properties": [no_categorization_material],
            "synthesis": None
        } for no_categorization_material in no_categorization_materials]

    # Use recursive_hierarchical_grouping to group the string keys
    grouped = recursive_hierarchical_grouping(descriptions_for_grouping, experimental_section)

    results = []
    # concurrent version
    def extract_single_process(rep_desc):
        process: Processing = agent.invoke({
            "material_description": rep_desc,
            "process_format": processing_format,
            "text": experimental_section
        }).record
        if process:
            process = SafeDumpProcessing.model_validate(process)
        return process
    reprs = [group['representative_processing_description'] for group in grouped]
    repr_to_process = dict(zip(reprs, run_concurrently(extract_single_process, reprs)))
    for group in grouped:
        all_objs = []
        for key in group["descriptions"]:
            all_objs.extend(desc_to_obj[key])
        represented_description = group['representative_processing_description']
        process = repr_to_process[represented_description]
        results.append({
            "representative": represented_description,
            "properties": all_objs,
            "synthesis": process
        })
    # append no categorization properties
    for m in no_categorization_materials:
        results.append({
            "representative": None,
            "properties": [m],
            "synthesis": None
        })
    return results


def extract_property(input_, agent: RunnableSequence) -> list[BaseModel]:
    r = agent.invoke(input_).records
    if not r:
        return []
    return r

class PaperResult(BaseModel):
    properties: Union[list[BaseModel], None]
    synthesis: Union[SafeDumpProcessing, None, Processing, ProcessingWithSymbol]

    def model_dump(self, **kwargs):
        if self.properties and self.synthesis is None:
            return {
                "properties": [{prop.__class__.__name__: prop.model_dump(**kwargs)} for prop in self.properties],
                "synthesis": None
            }
        if self.properties and self.synthesis:
            return {
                "properties": [{prop.__class__.__name__: prop.model_dump(**kwargs)} for prop in self.properties],
                "synthesis": self.synthesis.model_dump(**kwargs)
            }
        else:
            return super().model_dump(**kwargs)

def ensure_valid_dict(d):
    return {k: v for k, v in d.items() if v is not None}
    
def extract_contextualized_main(paragraphs: list[Paragraph], paragraphs_reconstr: dict[str, ParagraphExtend], agents: dict[str, RunnableSequence], formatted_func: Callable, save_to: str) -> list[ParagraphExtend]:
    """Main function for contextualized extraction"""
    # check existence of synthesis paragraph
    synthesis_paras = get_synthesis_paras(paragraphs)
    if not synthesis_paras:
        merge_func = merge_without_process
    else:
        merge_func = merge_with_process

    # extract properties
    # user input paragraphs dict may contain None value (consider situation when specific property is absent for an article), we should remove such value
    paragraphs_reconstr = ensure_valid_dict(paragraphs_reconstr)
    properties = [key for key in paragraphs_reconstr.keys() if key != "synthesis"]
    if not properties:
        return
    args = [({"text": paragraphs_reconstr[property].page_content}, agents[property]) for property in properties]
    results = run_concurrently(extract_property, args)
    results_flattened = [item for sublist in results for item in sublist]
    if not results_flattened:
        return

    # since the result returns in order, we can combine them
    results_dict = dict(zip(properties, results))
    for k in results_dict:
        paragraphs_reconstr[k].set_data(results_dict[k])

    # extract synthesis routes
    if synthesis_paras:
        experimental_section = paragraphs_reconstr['synthesis'].page_content
        formatted_instruction = formatted_func(experimental_section)
        results_with_processing = merge_func(results_flattened, formatted_instruction, experimental_section, agents['synthesis'])
        paper_results = [PaperResult(**result) for result in results_with_processing]
        # set synthesis routes for documents
        synthesis_routes = [result['synthesis'] for result in results_with_processing]
        paragraphs_reconstr['synthesis'].set_data(synthesis_routes)
    else:
        results_without_processing = merge_func(results_flattened)
        paper_results = [PaperResult(**result) for result in results_without_processing]


    # merge results and save
    doi = list(paragraphs_reconstr.values())[0].metadata.get('doi', None)
    dump_paper_results(
        [p.model_dump() for p in paper_results],
        doi,
        save_to
    )
    paras_save = list(paragraphs_reconstr.values())
    return paras_save


def extract_isolated_main(paragraphs_reconstr: dict[str, ParagraphExtend], agents: dict[str, RunnableSequence], formatted_func: Callable, save_to: str) -> list[ParagraphExtend]:
    paragraphs_reconstr = ensure_valid_dict(paragraphs_reconstr)
    properties = [key for key in paragraphs_reconstr.keys() if key != "synthesis"]
    args = [({"text": paragraphs_reconstr[property].page_content}, agents[property]) for property in properties]
    if (syn_para:=paragraphs_reconstr.get('synthesis', None)):
        experimental_section = paragraphs_reconstr['synthesis'].page_content
        formatted_instruction = formatted_func(experimental_section)
        args.append(({
            "text": syn_para.page_content,
            "process_format": formatted_instruction
        }, agents['synthesis']))
    results = run_concurrently(extract_property, args) # a list of lists, with each list contains serval dictionaries
    # since the result returns in order, we can combine them
    results_dict = dict(zip(properties, results))
    for k in results_dict:
        paragraphs_reconstr[k].set_data(results_dict[k])
    doi = paragraphs_reconstr[properties[0]].metadata.get('doi', None)
    merge_output_iso(results_dict, doi, save_to)
    paras_save = list(paragraphs_reconstr.values())
    return paras_save

def merge_output_iso(output, doi, save_to):
    new_output = []
    # no processing case
    if not output.get('synthesis', []):
        properties = []
        for value in output.values():
            properties.extend(value)
        symbol_dict = {}
        for prop in properties:
            symbol = prop.symbol
            if symbol:
                if symbol in symbol_dict:
                    symbol_dict[symbol].append(prop)
                else:
                    symbol_dict[symbol] = [prop]
            else:
                new_output.append(PaperResult(properties=[prop], synthesis=None))
        for props in symbol_dict.values():
            new_output.append(PaperResult(properties=props, synthesis=None))
    else:
        properties = []
        for k,v in output.items():
            if k != 'synthesis':
                properties.extend(v)
        synthesis_list = output['synthesis']
        for prop in properties:
            symbol = prop.symbol
            if symbol:
                if symbol in symbol_dict:
                    symbol_dict[symbol].append(prop)
                else:
                    symbol_dict[symbol] = [prop]
            else:
                new_output.append(PaperResult(properties=[prop], synthesis=None))
        for symbol, props in symbol_dict.items(): 
            according_synthesis = None
            for synthesis in synthesis_list:
                if synthesis.symbol == symbol:
                    according_synthesis = synthesis
                    break
            if according_synthesis:
                new_output.append(PaperResult(properties=props, synthesis=according_synthesis))
            else:
                new_output.append(PaperResult(properties=props, synthesis=None))
    dumps = [r.model_dump() for r in new_output]
    dump_paper_results(dumps, doi, save_to)


def merge_output(output, doi, save_to) -> list[PaperResult]:
    """merge those with same processing into one result"""
    # [(record, processing), ...]
    process_record_dict = {}
    process_pydantic_dict = {}
    without_process = []
    for record, processing in output:
        if processing is None:
            without_process.append(PaperResult(properties=[record], synthesis=None))
            continue
        
        key = processing.model_dump_json(warnings='none')
        if key in process_record_dict:
                process_record_dict[key].append(record)
                continue
        process_pydantic_dict[key] = processing
        process_record_dict[key] = [record]
    keys = process_pydantic_dict.keys()
    results = [PaperResult(properties=process_record_dict[key], synthesis=process_pydantic_dict[key]) for key in keys]
    results.extend(without_process)
    dumps = [r.model_dump() for r in results]
    dump_paper_results(dumps, doi, save_to)
    return results


_json_write_lock = threading.Lock()

def dump_paper_results(results, doi, jsonl_path):
    """
    Thread-safe write of paper_results to a JSONL file, grouping all results with the same DOI as one object.
    Args:
        results: list of dicts with 'doi' field (all with same doi per call)
        jsonl_path: path to output JSONL file
    """
    if not results:
        return

    if not doi:
        return

    obj = {
        "doi": doi,
        "results": results
    }

    with _json_write_lock:
        # Always append the new obj to the file
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
