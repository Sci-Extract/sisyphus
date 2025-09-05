from typing import Callable

from langchain_core.runnables import RunnableSequence

from sisyphus.chain.paragraph import ParagraphExtend, Paragraph
from sisyphus.strategy.llm_models import categorize_agent
from sisyphus.strategy.contextualized import extract_contextualized_main, extract_isolated_main
from sisyphus.strategy.utils import get_synthesis_paras, build_process_agent_contextualized, build_process_agent_isolated, build_property_agent, build_result_model_contextualized , build_result_model_isolated, build_process_model_contexualized
from sisyphus.strategy.pydantic_models_general import Processing, ProcessingWithSymbol, MaterialDescriptionBase, MaterialWithSymbol, Material


def extract_main(paragraphs: list[Paragraph], save_to: str, reconstruct_paragraph: Callable, property_agents_d: dict[str, list[RunnableSequence]], formatted_func: Callable, synthesis_agent: RunnableSequence, categorize_agent = categorize_agent) -> list[ParagraphExtend]:
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
    # category_labeled = categorize_agent(text=syn_paras.page_content).output
    paragraphs_reconstr = reconstruct_paragraph(paragraphs)
    #contextualized
    # if not category_labeled:
    # extract_contextualized_main(
    #     paragraphs_reconstr=paragraphs_reconstr,
    #     property_agents_d=property_agents_d,
    #     formatted_func=formatted_func,
    #     synthesis_agent=synthesis_agent,
    #     save_to=save_to
    # )
    # parawi 
    # else:

    
class ExtractStrategy:
    def __init__(self, reconstruct_paragraph_context_func, reconstruct_paragraph_isolate_func, formatted_func, categorize_agent, pydantic_models_dict, save_to):
        self.categorize_agent = categorize_agent
        self.formatted_func = formatted_func
        self.pydantic_models_dict = pydantic_models_dict
        self.reconstr_con = reconstruct_paragraph_context_func
        self.reconstr_iso = reconstruct_paragraph_isolate_func
        self.save_to = save_to
        self.contextualized = 'contextualized'
        self.isolated = 'isolated'
        self.agents = {'contextualized': {}, 'isolated': {}}

    def build(self, prompt_config, chat_model):
        """{'contextualized': {'property_1': (system_message, user_message), ...}, 'isolated': {...}}"""
        if prompt_config:
            for property in prompt_config[self.contextualized]:
                if property != 'synthesis':
                    self.agents[self.contextualized][property] = build_property_agent(
                        prompt_config[self.contextualized][property][0],
                        prompt_config[self.contextualized][property][1],
                        build_result_model_contextualized(
                            property,
                            self.pydantic_models_dict[property].__doc__,
                            self.pydantic_models_dict[property], MaterialDescriptionBase
                        ),
                        chat_model
                    )
                else:
                    self.agents[self.contextualized][property] = build_process_agent_contextualized(
                        prompt_config[self.contextualized][property][0],
                        prompt_config[self.contextualized][property][1],
                        build_process_model_contexualized(
                            property,
                            self.pydantic_models_dict[property].__doc__,
                            self.pydantic_models_dict[property], Material
                        ),
                        chat_model
                    )
            for property in prompt_config[self.isolated]:
                if property != 'synthesis':
                    self.agents[self.isolated][property] = build_property_agent(
                        prompt_config[self.isolated][property][0],
                        prompt_config[self.isolated][property][1],
                        build_result_model_isolated(
                            property,
                            self.pydantic_models_dict[property].__doc__,
                            self.pydantic_models_dict[property], MaterialWithSymbol
                        ),
                        chat_model
                    )
                else:
                    self.agents[self.isolated][property] = build_process_agent_isolated(
                        prompt_config[self.isolated][property][0],
                        prompt_config[self.isolated][property][1],
                        build_result_model_isolated(
                            property,
                            self.pydantic_models_dict[property].__doc__,
                            self.pydantic_models_dict[property], MaterialWithSymbol
                        ),
                        chat_model
                    )
        else:
            raise ValueError("Prompt config is required to build agents")
            

    # def __call__(self, paragraphs: list[Paragraph]):
    #     syn_paras = ParagraphExtend.from_paragraphs(get_synthesis_paras(paragraphs))
    #     # context mode
    #     if syn_paras:
    #         labeled = categorize_agent(text=syn_paras.page_content).output
    #         # only run below if materials are not clearly identified
    #         if not labeled:
    #             paragraphs_reconstructed = self.reconstr_con(paragraphs)
    #             return extract_contextualized_main(
    #                 paragraphs_reconstr=paragraphs_reconstructed,
    #                 agents=self.agents[self.contextualized],
    #                 formatted_func=self.formatted_func,
    #                 save_to=self.save_to
    #             )
    #     # otherwise, run isolated extraction
    #     paragraphs_reconstructed = self.reconstr_iso(paragraphs)
    #     return extract_isolated_main(
    #         paragraphs_reconstr=paragraphs_reconstructed,
    #         agents=self.agents[self.isolated],
    #         formatted_func=self.formatted_func,
    #         save_to=self.save_to
    #     )

    def __call__(self, paragraphs: list[Paragraph]):
        paragraphs_reconstructed = self.reconstr_con(paragraphs)
        return extract_contextualized_main(
            paragraphs_reconstr=paragraphs_reconstructed,
            agents=self.agents[self.contextualized],
            formatted_func=self.formatted_func,
            save_to=self.save_to
        )
           