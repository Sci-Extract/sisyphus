from langchain_core.documents import Document

from sisyphus.utils.helper_functions import render_docs


class Paragraph:
    def __init__(self, document: Document, id_ = None):
        self.document = document
        self.id = id_
        self.page_content = document.page_content
        self.metadata = document.metadata
        self.is_synthesis = False
        self.property_types = []
    
    def is_abstract(self):
        return True if self.metadata['sub_titles'] == 'Abstract' else False
    
    def is_table(self):
        return True if self.metadata['sub_titles'] == 'table' else False

    def has_property(self, property):
        return property in self.property_types
    
    def set_types(self, types):
        if isinstance(types, str):
            if types in self.property_types:
                return
            self.property_types.append(types)
        elif isinstance(types, list):
            for t in types:
                if t in self.property_types:
                    continue
                self.property_types.append(t)
        return

    def set_synthesis(self):
        self.is_synthesis = True
    

class ParagraphExtend(Paragraph):
    
    @classmethod
    def merge_paras(cls, paras, metadata, title, table_prefix='Tables:'):
        page_content = render_docs(paras, title, table_prefix)
        doc = Document(page_content, metadata=metadata)
        return cls(doc)