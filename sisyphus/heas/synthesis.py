import re
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from . import processing_template as pt
from . import processing_template_abbre as pt_abrev
from .synthesis_examples import examples_detect
from .embeddings import chroma_db, has_embedded, retrieve, match_subtitles, QUERY_SYN


class ClassifySyn(dspy.Signature):
    """assign topic to paragraphs of HEAs(high entropy alloys) papers. The topics include synthesis, characterization, and others.
    Note: a qualified synthesis paragraph should include the synthesis and processing of materials, including methods such as melting, casting, rolling, annealing, mechnical processes or additive manufacturing. be very strict about your decision."""
    paragraph: str = dspy.InputField()
    topic: Literal['synthesis', 'characterization', 'others'] = dspy.OutputField()


def label_syn_paras(docs, paras):
    """label the synthesis paragraphs in the paragraphs"""
    syn_pattern = re.compile(r'(experiment)|(preparation)|(method)', re.I)
    syn_titles = match_subtitles(docs, syn_pattern)
    source = docs[0].metadata['source']
    if source not in has_embedded:
        has_embedded.append(source)
        chroma_db.add_documents(docs)
    syn_docs = retrieve(chroma_db, source, QUERY_SYN, syn_titles, 5)
    syn_after_lm = []
    with ThreadPoolExecutor(5) as worker:
        futures = [worker.submit(dspy.ChainOfThought(ClassifySyn), paragraph=candidate.page_content) for candidate in syn_docs]
        future_doc = dict(zip(futures, syn_docs))
        for future in as_completed(futures):
            if future.result().topic == 'synthesis':
                syn_after_lm.append(future_doc[future])
    for syn_doc in syn_after_lm:
        for para in paras:
            if para.document == syn_doc:
                para.set_synthesis()

# categorize
templates = {k:v for k,v in pt.__dict__.items() if not k.startswith('__')}
# names = list(templates.keys())
templates_abrev = {k:v for k,v in pt_abrev.__dict__.items() if not k.startswith('__')}
names = list(templates.keys())

detect_processes_system_message = f"""Task: Find all synthesis processes mentioned in the given text. Each identified process must exactly match one of the elements in this list: {names}

Requirements:

Only return processes that appear in the provided list
Contain cooling if quenching/cooling is mentioned
Do not infer or fabricate processes not explicitly mentioned in the text
Return an empty list if no matching processes are found
Each process should be returned using its standard name from the list (e.g., return "aging" even if text says "precipitation hardening")"""

class DetectProcesses(dspy.Signature):
    paragraph: str = dspy.InputField()
    processes: list[str] = dspy.OutputField()


DetectProcesses.__doc__ = detect_processes_system_message
# predictor = dspy.LabeledFewShot().compile(dspy.Predict(DetectProcesses), trainset=examples_detect)
predictor = dspy.Predict(DetectProcesses)

def format_synthesis_prompt(processes):
    templates_string = ""
    for type_ in processes:
        if type_ in templates:
            templates_string += f'- {templates[type_]}\n'
    prompt = f"""For the "processes" field in the results, here are guideline for the output format:
- processes field must be a list of dictionaries. {{"processes": list}}
- Each dictionary consists of {{"process_method": str, information (different according to process method)}}

Here are predefined templates for some processes you must follow:
{templates_string} 

Dynamic process handling specification:
- When encountering a process that does not conform to predefined templates:
- Construct a flexible dictionary to represent the process.
- Ensure the dictionary maintains a consistent and extensible structure.
- MUST include a mandatory "process_method" key.
- Keys should be descriptive and represent the nature or type of the parameter.
e.g.
    {{
      "process_method": "custom_heat_treatment",
      "temperature": "500°C",
      "duration": "2 hours"
    }}

"""
    prompt_with_no_predefined = """For the "processes" field in the results, here are guideline for the output format:
- processes field must be a list of dictionaries. {{"processes": list}}
- Each dictionary consists of {{"process_method": str, information (different according to process method)}}

Dynamic process handling specification:
- Construct a flexible dictionary to represent the process.
- Ensure the dictionary maintains a consistent and extensible structure.
- MUST include a mandatory "process_method" key.
- Keys should be descriptive and represent the nature or type of the parameter.
e.g.
    {{
      "process_method": "custom_heat_treatment",
      "temperature": "500°C",
      "duration": "2 hours"
    }}

"""
    if not templates_string:
        return prompt_with_no_predefined

    return prompt

SYN_PROMPT_SIMPLE = """For the "steps" field in the results, here are guideline for the output format:
- Steps field must be a list of processing steps in JSON format.
- Steps should be ordered chronologically as they occur in the synthesis process.
- Steps must only include synthesis and processing methods directly related to the material's fabrication not characterization or testing.
- For fields in steps, if they are quantitive value, you should provide unit along with it if possible.
- You do not need to use all templates; only include those that are relevant to the synthesis described.

** Required templates **
{formatted_string}
** Dynamic process handling specification: **
- If a synthesis step does not fit any predefined template, create a custom entry where:
  - The key should be the actual name of the process/method being used (e.g., "rotary swaging", "plasma treatment", "electrodeposition")
  - Include relevant parameters as nested values

Example format:
{{
    "<actual_process_name>": {{
        "parameter1": "value1",
        "parameter2": "value2"
    }}
}}
"""

def format_synthesis_prompt_abbrev(processes):
    """format synthesis prompt to instruct LLM output more consistent synthesis route"""
    processes_filtered = [process for process in processes if process in templates_abrev]
    if not processes_filtered:
        return SYN_PROMPT_SIMPLE 
    processes_string = ', '.join(processes_filtered)
    prompt = SYN_PROMPT_SIMPLE + """\nYou may include these processes in your response:\n- {processes_string}"""
    return prompt.format(processes_string=processes_string)

def format_synthesis_prompt_str(processes):
    """format synthesis prompt to instruct LLM output more consistent synthesis route with string formatted instruction"""
    processes_filtered = [process for process in processes if process in templates]
    formatted_string = ''
    for process in processes_filtered:
        formatted_string += f'- {templates[process]}\n'
    prompt = SYN_PROMPT_SIMPLE.format(formatted_string=formatted_string)
    return prompt

def get_synthesis_prompt(text, lm=dspy.LM('openai/gpt-4.1')):
    """return the formatted prompt information for synthesis extraction"""
    if not text:
        return format_synthesis_prompt_str([])
    with dspy.context(lm=lm):
        types = predictor(paragraph=text).processes
    synthesis_prompt = format_synthesis_prompt_str(types)
    return synthesis_prompt

def get_synthesis_prompt_all():
    """return all information about the synthesis template"""
    return format_synthesis_prompt_str(names)
