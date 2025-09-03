# Implementation of contexualized extraction: Frist extract properties separately then link it with correponds synthesis routes via description of that material.
# TODO  1. add para-wise extraction
import json
import threading
import os
import warnings
from typing import Optional, Callable, Union

import dspy
from pydantic import BaseModel
from langchain_core.runnables import RunnableSequence

from sisyphus.utils.helper_functions import run_concurrently
from sisyphus.chain.paragraph import ParagraphExtend
from sisyphus.strategy.pydantic_models_general import MaterialDescriptionBase, Processing, SafeDumpProcessing, ProcessingWithSymbol


warnings.filterwarnings('ignore', category=UserWarning, module='pydantic') # the case that we convert json string to python object trigger pydantic warning

def extract_process(property_: MaterialDescriptionBase, processing_format, material_process_dict, experimental_section: str, agent) -> SafeDumpProcessing:
    composition, description, refered = property_.composition, property_.description, property_.refered
    if refered:
        # this is a reference, we do not extract processing route
        return
    if not any([composition, description]):
        # this is not a valid extraction
        return
    if (t:=(composition, description)) in material_process_dict:
        return material_process_dict[t]
    # find same reference if possible, based on composition, description
    key_index = find_same_reference(composition, description, material_process_dict)
    if key_index is not None:
        process = material_process_dict[list(material_process_dict.keys())[key_index]]
        return process
    # otherwise, extract processing route
    process: Processing = agent.invoke(
        {
            "material_description": f"{composition}: {description}",
            "process_format": processing_format,
            "text": experimental_section
        }
    ).record
    if process:
        process = SafeDumpProcessing.model_validate(process)
    material_process_dict.update({(composition, description): process})
    return process

def print_comp_description(material_process_dict):
    table = "| Index | Composition | Description |\n|-------|-------------|-------------|\n"
    for idx, ((composition, description), _) in enumerate(material_process_dict.items()):
        table += f"| {idx} | {composition or ''} | {description or ''} |\n"
    return table

class FindSameReference(dspy.Signature):
    """Find which row in the table refered to the same material as provided composition and description.
    Consider both composition and description.
    Be very strict, only return the index if you are sure, otherwise return None."""
    table: str = dspy.InputField(description="Table of material composition and description, in markdown format")
    composition: str = dspy.InputField(description="The composition of the material, e.g., 'Mn0.2CoCrNi'")
    description: str = dspy.InputField(description="The description of the material, e.g., 'as-cast', 'annealed at 900C'")
    index: Optional[int] = dspy.OutputField(description="The index of the row")
get_same_reference_agent = dspy.ChainOfThought(FindSameReference)

def find_same_reference(composition: str, description: str, material_process_dict):
    if len(material_process_dict) == 0:
        return None
    index = get_same_reference_agent(
        table=print_comp_description(material_process_dict),
        composition=composition,
        description=description
    ).index
    return index

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
    
def extract_contextualized_main(paragraphs_reconstr: dict[str, ParagraphExtend], agents: dict[str, RunnableSequence], formatted_func: Callable, save_to: str) -> list[ParagraphExtend]:
    output = []
    # extract properties
    # user input paragraphs dict may contain None value (consider situation when specific property is absent for an article), we should remove such value
    paragraphs_reconstr = ensure_valid_dict(paragraphs_reconstr)
    properties = [key for key in paragraphs_reconstr.keys() if key != "synthesis"]
    args = [({"text": paragraphs_reconstr[property].page_content}, agents[property]) for property in properties]
    results = run_concurrently(extract_property, args) # a list of lists, with each list contains serval dictionaries
    # since the result returns in order, we can combine them
    results_dict = dict(zip(properties, results))
    for k in results_dict:
        paragraphs_reconstr[k].set_data(results_dict[k])

    # extract synthesis routes
    material_process_d = {}
    experimental_section = paragraphs_reconstr['synthesis'].page_content
    formatted_instruction = formatted_func(experimental_section)
    for records_property in results:
        for record in records_property:
            processing = extract_process(
                record,
                formatted_instruction,
                material_process_d,
                experimental_section,
                agents['synthesis']
            )
            paragraphs_reconstr['synthesis'].set_data(processing)
            output.append((record, processing))
    doi = paragraphs_reconstr['synthesis'].metadata.get('doi', None)
    merge_output(output, doi, save_to)
    paras_save = list(paragraphs_reconstr.values())
    return paras_save


def extract_isolated_main(paragraphs_reconstr: dict[str, ParagraphExtend], agents: dict[str, RunnableSequence], formatted_func: Callable, save_to: str) -> list[ParagraphExtend]:
    paragraphs_reconstr = ensure_valid_dict(paragraphs_reconstr)
    properties = [key for key in paragraphs_reconstr.keys() if key != "synthesis"]
    args = [({"text": paragraphs_reconstr[property].page_content}, agents[property]) for property in properties]
    if (syn_para:=paragraphs_reconstr.get('synthesis', None)):
        experimental_section = paragraphs_reconstr['synthesis'].page_content
        formatted_instruction = formatted_func(experimental_section)
        args.append({
            "text": syn_para.page_content,
            "process_format": formatted_instruction
        }, agents['synthesis'])
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
