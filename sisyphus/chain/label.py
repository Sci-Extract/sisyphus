import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from sisyphus.chain.chain_elements import BaseElement
from sisyphus.chain.paragraph import Paragraph
from sisyphus.utils.helper_functions import get_plain_articledb

class BaseLabeler:

    property: str = None
    regex_pattern: re.Pattern = None
    query: str = None
    llm_labeler = None
    
    def regex_label(self, paragraph: Paragraph):
        text = paragraph.page_content
        if self.regex_pattern is None or self.regex_pattern.search(text):
            return True
    
    def semantic_label(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        if self.query:
            raise NotImplementedError("Please implement the semantic_label method in subclasses")
        return paragraphs

    def llm_label(self, paragraph: Paragraph):
        if self.llm_labeler:
            raise NotImplementedError("Please implement the llm_label method in subclasses")
        return True

    def label(self, paragraphs: list[Paragraph]):
        semantic_candidates = self.semantic_label(paragraphs)

        regex_candidates = []
        for para in semantic_candidates:
            has_regex = self.regex_label(para)
            if has_regex:
                regex_candidates.append(para)

        for para in regex_candidates:
            llm_result = self.llm_label(para)
            if llm_result:
                para.set_types(self.property)

        return paragraphs
    
class Labeling(BaseElement):
    def __init__(self):
        self.labelers = []
    
    def add_labeler(self, labeler: BaseLabeler):
        self.labelers.append(labeler)
    
    def label(self, paragraphs: list[Paragraph]):
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda labeler: labeler.label(paragraphs), self.labelers)
        return paragraphs
    
    def invoke(self, docs: list[Document]):
        paragraphs = [Paragraph(doc, id_) for id_, doc in enumerate(docs)]
        labeled_paras = self.label(paragraphs)
        return labeled_paras

def save_labeled_paras_wrapper(database_name):
    labeled_database = get_plain_articledb(database_name)
    labeled_database.create_db()
    def save(paras):
        labeled_database.dump_state(paras)        
    return save
    