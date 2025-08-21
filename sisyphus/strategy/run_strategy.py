from typing import Callable

from langchain_core.runnables import RunnableSequence

from sisyphus.chain.paragraph import ParagraphExtend, Paragraph
from sisyphus.strategy.llm_models import categorize_agent
from sisyphus.strategy.contextualized import extract_contextualized_main
from sisyphus.strategy.utils import get_synthesis_paras


def extract_main(paragraphs: list[Paragraph], reconstruct_paragraph: Callable, property_agents_d: dict[str, list[RunnableSequence]], formatted_func: Callable, synthesis_agent: RunnableSequence, categorize_agent = categorize_agent, save_to = 'output.json') -> list[ParagraphExtend]:
    """
    extract_main extract paragraphs and tables from paragraph with two mode: para-wise or contextualized


    Args:
        paragraphs (list[Paragraph]): 
        reconstruct_paragraph (Callable): which takes paragraphs as input and output a dict which indicates the type of the paragraph. e.g. {"synthesis": `ParaExtend`, "strength": `ParaExtend`}
        property_agents_d (dict[str, list[RunnableSequence]]): a map of property to extract agent which takes text as input, use langchain agent
        formatted_func (Callable): which takes experimental section string as input and output formatted instruction to guide later synthesis extraction
        synthesis_agent (RunnableSequence): synthesis agent, currently support context
        categorize_agent (dspy.Predict, optional): used to decide between two mode. Defaults to categorize_agent.

    Returns:
        list[ParagraphExtend]: list of paragraphs with data set in the attribute `.data`
    """
    syn_paras = ParagraphExtend.from_paragraphs(get_synthesis_paras(paragraphs))
    category_labeled = categorize_agent(text=syn_paras.page_content).output
    paragraphs_reconstr = reconstruct_paragraph(paragraphs)
    if not category_labeled:
        extract_contextualized_main(
            paragraphs_reconstr=paragraphs_reconstr,
            property_agents_d=property_agents_d,
            formatted_func=formatted_func,
            synthesis_agent=synthesis_agent,
            save_to=save_to
        )
    
    