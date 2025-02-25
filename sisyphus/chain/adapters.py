# TODO: If we want to extract multi different data, we can use the selector to create label field in the docs metadata, so that it can be propogated to the extraction steps for routing of correspond data format.
from typing import Callable

from pydantic import BaseModel

from sisyphus.chain.chain_elements import BaseElement, Document, DocInfo
from sisyphus.chain.llm_modules import cems_post_process
from sisyphus.chain.customized_elements import customized_extractor
from sisyphus.utils.helper_functions import get_title_abs, render_docs, reorder_docs


class ParagraphSelectionAdapter(BaseElement):
    def __init__(self, selector, with_context=True, table_prefix='Tables:'):
        self.selector = selector
        self.with_context = with_context
        self.table_prefix = table_prefix
    
    def invoke(self, docs):
        docs_filtered = []
        title, abstracts = get_title_abs(docs)
        for doc in docs:
            if self.selector(doc):
                if self.with_context:
                    if not doc.metadata['sub_titles'] == 'Abstract':
                        metadata = doc.metadata
                        metadata.update({'type': 'paragraph-wise'})
                        doc = Document(page_content=render_docs(abstracts + [doc], title, self.table_prefix), metadata=metadata)
                docs_filtered.append(doc)
        if not docs_filtered:
            return None
        return {'docs_original': docs, 'docs_filtered': docs_filtered}
    

class ContextulizedSelectionAdapter(BaseElement):
    def __init__(self, selector, with_context=True, tabel_prefix='Tables:'):
        self.selector = selector
        self.with_context = with_context
        self.table_prefix = tabel_prefix
    
    def invoke(self, docs):
        title, abstracts = get_title_abs(docs)
        docs_filtered = self.selector(docs)
        if not docs_filtered:
            return
        for doc in docs_filtered:
            if doc not in docs:
                raise ValueError('The selected docs should be in the original docs')
        if self.with_context:
            docs_filtered.extend(abstracts)
        
        original_docs_ordered = list(zip(docs, range(len(docs))))
        docs_filtered_ordered = reorder_docs(original_docs_ordered, docs_filtered)
        metadata = docs[0].metadata
        metadata.update({'type': 'contextulized'})
        return Document(page_content=render_docs(docs_filtered_ordered, title, self.table_prefix), metadata=metadata)


class ParagraphExtractionAdapter(BaseElement):
    def __init__(self, extractor: Callable[[Document], list[BaseModel|dict]], mode, concurrent_number=5):
        self.extractor = extractor
        self.mode = mode
        self.concurrent_number = concurrent_number
    
    def invoke(self, d):
        docs = d.get('docs_filtered')
        if self.mode == 'async':
            docinfos = customized_extractor(self.extractor, self.mode, self.concurrent_number)(docs)
        elif self.mode == 'thread':
            docinfos =  customized_extractor(self.extractor, self.mode, self.concurrent_number)(docs)
        elif self.mode == 'normal':
            docinfos = customized_extractor(self.extractor, self.mode)(docs)
        else:
            raise ValueError('mode should be one of "async", "thread" or "normal"')
        if not docinfos:
            return None
        d.pop('docs_filtered')
        d.update({'docs_info': docinfos})
        return d
    
class ParagraphCemValidationAdapter(BaseElement):
    def __init__(self, cem_collector: Callable[[BaseModel | dict], list[str]], cem_updater: Callable[[BaseModel | dict, dict | None, list], BaseModel | None | dict], drop_rule=None, standard_rule=None):
        self.cem_collector = cem_collector
        self.drop_rule = drop_rule
        self.cem_updater = cem_updater
        self.standard_rule = standard_rule
    def invoke(self, d):
        docs = d.get('docs_original')
        docinfos = d.get('docs_info')
        cems = []
        updated_docinfos = []
        for docinfo in docinfos:
            for model in docinfo.info:
                cems.extend(self.cem_collector(model))
        cems = list(set(cems))
        acronym_dict, drops = cems_post_process(docs, cems, self.drop_rule, self.standard_rule)
        for docinfo in docinfos:
            updated_models = []
            for model in docinfo.info:
                updated_model = self.cem_updater(model, acronym_dict, drops)
                if updated_model:
                    updated_models.append(updated_model)
            if updated_models:
                updated_docinfos.append(DocInfo(doc=docinfo.doc, info=updated_models))
        if not updated_docinfos:
            return None
        return updated_docinfos
    