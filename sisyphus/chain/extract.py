from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from sisyphus.chain.chain_elements import BaseElement
from sisyphus.strategy.utils import get_paras_with_props, get_synthesis_paras
from sisyphus.chain.paragraph import Paragraph, ParagraphExtend


class BaseExtractor:
    def __init__(self, target_properties: list[str], context_properties: list[str], model: ChatOpenAI):
        """init with corresponded properties and chat model used for extraction. If use another chat model, make sure change extract method"""
        self.target_properties = target_properties
        self.context_properties = context_properties
        self.model = model
        self.output_model = None
        self.prompt = None
        self.lock = Lock()

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
            type=self.target_properties[0]
        ) # merge paragraphs and add a type attribute
        return [p]

    def extract(self, paragraph: ParagraphExtend) -> ParagraphExtend: 
        chain = self.prompt | self.model.with_structured_output(
            schema=self.output_model,
            method='json_schema'
        )
        results = chain.invoke(
            {
               'text': paragraph.page_content,
               'instruction': paragraph.instruction
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
        with self.lock:
            self.create_model_prompt(paragraphs)
            if not(self.output_model and self.prompt):
                raise RuntimeError('please implement the create_model_prompt method')
            regrouped_pragraphs = self.regroup(paragraphs)
            if not regrouped_pragraphs:
                return
            data_paragraphs = self.extract_parallel(regrouped_pragraphs)
            data_paragraphs = [p for p in data_paragraphs if p]
            return data_paragraphs
    
    def create_model_prompt(self, paragraphs: list[Paragraph]):
        """dynamically create output_model and prompt w.r.t input paragraphs, update `self.output_model` and `self.prompt`"""
        pass
    

class Extraction(BaseElement):
    def __init__(self, merge_func=None):
        self.extractors = []
        self.merge_func = merge_func

    def add_extractors(self, extractors: list[BaseExtractor]):
        self.extractors.append(extractors)

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
    