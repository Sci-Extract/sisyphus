# -*- coding:utf-8 -*-
'''
@File    :   archive.py
@Time    :   2024/04/21 20:51:27
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   load, parse article from html to `Document` object.
Store article into sql.
'''

from collections import namedtuple
from typing import AsyncIterator

import nltk
import aiofiles
import tiktoken
from bs4 import BeautifulSoup as bs
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


encoding = tiktoken.get_encoding('cl100k_base')
MetaData = namedtuple("MetaData", "source section title")

class ArticleLoader(BaseLoader):
    """convert article.html to langchain `Document` object"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]

    async def alazy_load(self) -> AsyncIterator[Document]:
        async with aiofiles.open(self.file_path, encoding='utf8') as file:
            doc = await file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find("title").text.strip('\n ')
        abstract = soup.css.select("div#abstract > p")[0]
        abstract_chunks = self.chunk_text(abstract.text.strip('\n '))
        for chunk in abstract_chunks:
            yield Document(page_content=chunk, metadata=MetaData(self.file_name, "abstract", title)._asdict())
        for section in self.get_sections(soup=soup, title=title):
            yield section
    
    def get_sections(self, soup: bs, title: str):
        """
        Get sections except abstract.
        """
        section = None
        for child in soup.find(id="sections"):
            if child.name == "h2":
                section = child.text.strip('\n ')
            elif child.name == "p":
                chunks = self.chunk_text(child.text.strip('\n '))
                for chunk in chunks:
                    yield Document(
                    page_content=chunk,
                    metadata=MetaData(self.file_name, section, title)._asdict()
                    )

    def chunk_text(self, text) -> list[str]:
        """
        Due to the input text is a paragraph, to preserve the semantic meanings per chunk.
        Chunking the text more than 400 tokens, meanwhile, prevent generating small chunks less than 200 tokens.
        The token range per chunk is 200 - 600. 
        """
        if len(encoding.encode(text)) <= 400:
            return [text]
        sentences = nltk.sent_tokenize(text)
        token_per_sent = [len(encoding.encode(sent)) for sent in sentences]
        chunked_texts = []
        accumulate_token = 0
        next_start_i = 0
        for i, token in enumerate(token_per_sent):
            if i == len(token_per_sent) - 1:
                chunked_texts.append(' '.join(sentences[next_start_i:]))
                break
            accumulate_token += token
            token_left = sum(token_per_sent[i:]) # yes, I intend to keep this token in the result
            
            if accumulate_token <= 400:
                continue
            if token_left >= 200:
                chunked_texts.append(' '.join(sentences[next_start_i: i]))
                next_start_i = i
                accumulate_token = token
            else:
                chunked_texts.append(' '.join(sentences[next_start_i:]))
                break
        return chunked_texts
        