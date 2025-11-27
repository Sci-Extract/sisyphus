from typing import Callable

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from sisyphus.chain import Paragraph, ParagraphExtend
from sisyphus.strategy.utils import get_paras_with_props, get_synthesis_paras

def extract_property(paragraphs: list[Paragraph], labels: str, extract_func: Callable, **kwargs) -> ParagraphExtend:
    """Extracts the specified property from a list of Paragraphs.

    Args:
        paragraphs (list[Paragraph]): List of Paragraph objects.
        labels (str): The property labels to extract.
    """
    if syn_paras:=get_synthesis_paras(paragraphs):
        target_paras = get_paras_with_props(paragraphs, labels)
        paragraph = ParagraphExtend.from_paragraphs(target_paras+syn_paras, **kwargs)
        paragraph.set_data(extract_func(paragraph))
        return paragraph

def extract_func_wrapper(model: ChatOpenAI, prompt: ChatPromptTemplate, output_model: BaseModel):
    chain = prompt | model.with_structured_output(output_model, method='json_schema')
    def wrapper(input_):
        result = chain.invoke(input_)
        return result.records
    return wrapper

    