# Implementation of contexualized extraction: Frist extract properties separately then link it with correponds synthesis routes via description of that material.
from typing import Optional, Callable

import dspy
from dspy import Predict
from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from sisyphus.utils.helper_functions import run_concurrently
from sisyphus.chain.paragraph import Paragraph, ParagraphExtend
from sisyphus.strategy.pydantic_models_general import MaterialDescriptionBase, Processes


def extract_process(property: MaterialDescriptionBase, processing_format, material_process_dict, experimental_section: str, agent) -> list[Processes]:
    composition, description, refered = property.composition, property.description, property.refered
    if refered:
        # this is a reference, we do not extract processing route
        return
    if not any([composition, description]):
        # this is not a valid extraction
        return
    if t:=(composition, description) in material_process_dict:
        return material_process_dict[t]
    # find same reference if possible, based on composition, description
    key_index = find_same_reference(composition, description, material_process_dict)
    if key_index is not None:
        process = material_process_dict[list(material_process_dict.keys())[key_index]]
        return process
    # otherwise, extract processing route
    process = agent.invoke(
        {
            "material_description": f"{composition}: {description}",
            "process_format": processing_format,
            "text": experimental_section
        }
    )
    if not process:
        return []
    return process

def print_comp_description(material_process_dict):
    table = "| Index | Composition | Description |\n|-------|-------------|-------------|\n"
    for idx, ((composition, description), _) in enumerate(material_process_dict.items()):
        table += f"| {idx} | {composition or ''} | {description or ''} |\n"
    return table

class FindSameReference(dspy.Signature):
    """Find which row in the table refered to the same material as provided composition and description.
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
    )
    return index

def extract_property(input_, agent: RunnableSequence) -> list[BaseModel]:
    r = agent.invoke(input_).records
    if not r:
        return []
    return r

class PaperResult(BaseModel):
    properties: list[BaseModel] 
    synthesis: list[dict]
    
def extract_contextualized_main(paragraphs_reconstr: dict[str, ParagraphExtend], property_agents_d: dict[str, list[RunnableSequence]], formatted_func: Callable, synthesis_agent: RunnableSequence) -> list[ParagraphExtend]:
    output = []
    # extract properties
    properties = [key for key in paragraphs_reconstr.keys() if key != "synthesis"]
    args = [({"text": paragraphs_reconstr[property].page_content}, property_agents_d[property]) for property in properties]
    results = run_concurrently(extract_property, args) # a list of lists, with each list contains serval dictionaries
    # since the result returns in order, we can
    results_dict = dict(zip(properties, results))
    for k, para in paragraphs_reconstr.items():
        para.set_data(results_dict[k])

    # extract synthesis routes
    paras_store_exp_results = []
    material_process_d = {}
    experimental_section = paragraphs_reconstr['synthesis'].page_content
    formatted_instruction = formatted_func(experimental_section)
    for records_property in results:
        for record in records_property:
            syn_routes = extract_process(
                record,
                formatted_instruction,
                material_process_d,
                experimental_section,
                synthesis_agent
            )
            composition, description = record.composition, record.description
            paras_store_exp_results.append(
                ParagraphExtend.from_paragraphs(
                    [paragraphs_reconstr['synthesis']],
                    composition=composition,
                    description=description,
                ).set_data(syn_routes)
            )

            output.append(PaperResult(
                properties=record,
                synthesis=syn_routes
            ))
    return paragraphs_reconstr.values().extend(paras_store_exp_results)

# def reconstruct_paragraphs(paragraphs: list[Paragraph]) -> dict[str, ParagraphExtend]:
#     # how you gonna reconstruct the paragraphs for your extraction targets
#     # example implementation ...
#     strentgh_paras = ParagraphExtend.from_paragraphs([p for p in paragraphs if p.has_property('strength')], type='strength')
#     phase_paras = ParagraphExtend.from_paragraphs([p for p in paragraphs if p.has_property('phase')], type='phase')
#     synthesis_paras = ParagraphExtend.from_paragraphs([p for p in paragraphs if p.has_property('synthesis')], type='synthesis')
#     return {
#         "strength": strentgh_paras,
#         "phase": phase_paras,
#         "synthesis": synthesis_paras
#     }


# Input are labled pararaphs, tables, remember to save them to local storage.
# First reconstruct them to produce properties, synthesis paragraphs
# Then decide context or not extration
# for every property, extract it (parallel)
# and link it with the corresponding processing route
# finnaly merge the results if they are same material under same processing route
