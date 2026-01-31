from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from sisyphus.chain.chain_elements import BaseElement
from sisyphus.strategy.utils import get_paras_with_props, get_synthesis_paras
from sisyphus.chain.paragraph import Paragraph, ParagraphExtend


class BaseExtractor:

    target_properties: list[str] = []
    context_properties: list[str] = []
    model: ChatOpenAI = None  # chat model

    def regroup_(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        target_paras = []
        if 'synthesis' in self.target_properties:
            target_paras.extend(get_synthesis_paras(paragraphs))
        target_paras.extend(get_paras_with_props(paragraphs, *self.target_properties)) 
        if not target_paras:
            return
        context_paras = get_paras_with_props(paragraphs, *self.context_properties)
        return target_paras + context_paras
    
    def regroup(self, paragraphs: list[Paragraph]) -> list[ParagraphExtend]:
        """regroup paragraphs with respect to property, default to merge paragraphs to one paragraph object""" 
        p = ParagraphExtend.from_paragraphs(
            self.regroup_(paragraphs),
            inherit_properties=True, # merged paragraph should contain all paragraph's property types
            type=self.target_properties[0]
        ) # merge paragraphs and add a type metadata
        return [p]

    def extract(self, paragraph: ParagraphExtend) -> ParagraphExtend: 
        chain = paragraph.prompt_template | self.model.with_structured_output(
            schema=paragraph.pydantic_model,
            method='json_schema'
        )
        if paragraph.prompt_vars_dict:
            extra_kwargs = paragraph.prompt_vars_dict
        else:
            extra_kwargs = {}
        results = chain.invoke(
            {
               'text': paragraph.page_content,
               **extra_kwargs
            }
        )
        if records:=getattr(results, 'records'):
            paragraph.set_data(records)
            return paragraph
        paragraph.set_data(results)
        return paragraph
    
    def extract_parallel(self, paragraphs: list[ParagraphExtend]):
        with ThreadPoolExecutor(max_workers=5) as executor:
            paragraphs = list(filter(None, paragraphs))
            para_iter = executor.map(self.extract, paragraphs)
            paras = [p for p in para_iter]
        return paras
    
    def regroup_then_extract(self, paragraphs: list[Paragraph]):
        regrouped_pragraphs = self.regroup(paragraphs)
        self.create_model_prompt(regrouped_pragraphs)
        for paragraph in regrouped_pragraphs:
            if not(paragraph.prompt_template and paragraph.pydantic_model):
                raise RuntimeError('please set prompt_template and pydantic_model for each paragraph before extraction!')
        if not regrouped_pragraphs:
            return
        data_paragraphs = self.extract_parallel(regrouped_pragraphs)
        data_paragraphs = [p for p in data_paragraphs if p]
        return data_paragraphs
    
    def create_model_prompt(self, paragraphs: list[ParagraphExtend]):
        """dynamically create output_model and prompt w.r.t merged paragraphs. Update `paragraph.output_model`, `paragraph.prompt_template` and `paragraph.prompt_vars_dict` for each paragraph
        - Note: invoke paragraph.set_pydantic_model and paragraph.set_prompt in this function
        """
        pass
    

class Extraction(BaseElement):
    def __init__(self, merge_func=None):
        self.extractors = []
        self.merge_func = merge_func

    def add_extractor(self, extractor: BaseExtractor):
        self.extractors.append(extractor)

    def extract(self, paragraphs: list[Paragraph]):
        with ThreadPoolExecutor(max_workers=5) as executor:
            extractor_iter = executor.map(lambda ext: ext.regroup_then_extract(paragraphs), self.extractors)
            all_data_paras = []
            for data_paras in extractor_iter:
                if data_paras:
                    all_data_paras.extend(data_paras)
        return all_data_paras
    
    def invoke(self, paragraphs: list[Paragraph]):
        data_paragraphs = self.extract(paragraphs)
        if self.merge_func:
            merge_results = self.merge_func(paragraphs, data_paragraphs) # custom merge function
        return list(filter(None, data_paragraphs))
    