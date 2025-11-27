from langchain_core.documents import Document


class Paragraph:
    def __init__(self, document: Document, id_ = None):
        self.document = document
        self.id = id_
        self.page_content = document.page_content
        self.metadata = document.metadata
        self.is_synthesis = False
        self.property_types = []
        self.data = []
    
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

    def update_metadata(self, key, value):
        self.metadata[key] = value

    @classmethod
    def from_labeled_document(cls, labeled_document: Document, id_):
        # check labels field in metadata
        assert 'labels' in labeled_document.metadata, "labeled_document should have 'labels' field in metadata"
        labels = labeled_document.metadata['labels']
        instance = cls(Document(labeled_document.page_content, metadata={k: v for k, v in labeled_document.metadata.items() if k != 'labels'}), id_=id_)
        if 'is_synthesis' in labels:
            instance.set_synthesis()
        if 'property_types' in labels:
            instance.set_types(labels['property_types'])
        return instance

    def set_data(self, data):
        if not data:
            return self
        if type(data) is not list:
            data = [data]
        data = [d for d in data if d]
        self.data.extend(data)
        return self


class ParagraphExtend(Paragraph):
    """This class does not inherited paragraphs type attribute! You should set it again if needed."""
    metadata_keys_inherited = ['source', 'doi', 'title']
    @classmethod
    def merge_paras(cls, paras, metadata, title, table_prefix='Tables:'):
        """Merge multiple paragraphs into one paragraph with title."""
        from sisyphus.utils.helper_functions import render_docs
        page_content = render_docs(paras, title, table_prefix)
        doc = Document(page_content, metadata=metadata)
        return cls(doc)
    
    @classmethod
    def from_paragraphs(cls, paras, **metadata):
        from sisyphus.utils.helper_functions import render_docs
        if not paras:
            return
        new_metadata = {k: v for k, v in paras[0].metadata.items() if k in cls.metadata_keys_inherited}
        new_metadata.update(metadata)
        page_content = render_docs(paras, new_metadata['title'])
        doc = Document(page_content, metadata=new_metadata)
        return cls(doc)
    