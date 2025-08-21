from typing import Optional

from pydantic import create_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from sisyphus.chain.paragraph import Paragraph

def build_result_model_contextualized(name: str, model_document: str, *bases):
    r = create_model(name, __base__=bases, __doc__=model_document)
    assert all(f in r.model_fields.keys() for f in ['composition', 'description', 'refered']), "Fields should include composition, description and refered"
    return create_model('Records', records=(Optional[list[r]], ...))

def build_property_agent(system_message, user_message, pydantic_model, chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_message),
            ('user', user_message)
        ]
    )
    input_vars = prompt.input_variables
    assert set(['text']) == set(input_vars), f"Input variables should be ['text'], but got {input_vars}"
    agent = prompt | chat_model.with_structured_output(pydantic_model)
    return agent

def build_process_agent(system_message, user_message, pydantic_model, chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_message),
            ('user', user_message)
        ]
    )
    input_vars = prompt.input_variables
    assert set(['text', 'material_description', 'process_format']) == set(input_vars), f"Input variables should be ['text', 'material_description', 'process_format'], but got {input_vars}"
    agent = prompt | chat_model.with_structured_output(pydantic_model)
    return agent

def format_processes(processes: list[str], processes_format_dict) -> str:
    format_string = ""
    for process in processes:
        if process not in processes_format_dict:
            continue
        format_string += f"{processes_format_dict[process]}\n"
    return format_string.strip()

def get_synthesis_paras(paras: list[Paragraph]):
    return [para for para in paras if para.is_synthesis]

def get_paras_with_props(paras: list[Paragraph], *property):
    return [para for para in paras if len(set(para.property_types).intersection(set(property))) >= 1]