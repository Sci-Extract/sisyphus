# -*- coding:utf-8 -*-
"""
@File    :   archive.py
@Time    :   2024/04/21 20:51:27
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   load, parse article from html to `Document` object.
"""

import os
import glob
from collections import namedtuple
from typing import AsyncIterator, Iterator

import nltk
import aiofiles
import tiktoken
from bs4 import BeautifulSoup as bs
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain.pydantic_v1 import (
    BaseModel,
    create_model
)
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

encoding = tiktoken.get_encoding('cl100k_base')

# region loader
class Loader(BaseLoader):
    """base loader"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata: BaseModel = None # subclass must define their own metadata

class ArticleLoader(Loader):
    # TODO: load table
    """convert article.html to langchain `Document` object
    
    presently not considering table
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_name = file_path.split(os.sep)[-1]
        self.metadata = create_model('MetaData', source=(str, ...), section=(str, ...), title=(str, ...))

    async def alazy_load(self) -> AsyncIterator[Document]:
        async with aiofiles.open(self.file_path, encoding='utf8') as file:
            doc = await file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find('title').text.strip('\n ')
        try:
            abstract = soup.css.select('div#abstract > p')[0]
        except IndexError:
            print(self.file_name)
            return
        abstract_chunks = self.chunk_text(abstract.text.strip('\n '))
        for chunk in abstract_chunks:
            yield Document(
                page_content=chunk,
                metadata=dict(self.metadata(source=self.file_name, section='abstract', title=title)),
            )
        for section in self.get_sections(soup=soup, title=title):
            yield section

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding='utf8') as file:
            doc = file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find('title').text.strip('\n ')
        try:
            abstract = soup.css.select('div#abstract > p')[0]
        except IndexError:
            print(self.file_name)
            return
        abstract_chunks = self.chunk_text(abstract.text.strip('\n '))
        for chunk in abstract_chunks:
            yield Document(
                page_content=chunk,
                metadata=dict(self.metadata(source=self.file_name, section='abstract', title=title)),
            )
        for section in self.get_sections(soup=soup, title=title):
            yield section

    def get_sections(self, soup: bs, title: str):
        """
        Get rest sections.
        TODO: better use nested section name, e.g. Experimental/<sub_section>
        """
        section = (
            'abstract'  # a few papers have abstract which more than one para
        )
        for child in soup.find(id='sections'):
            if child.name == 'h2':
                section = child.text.strip('\n ')
            elif child.name == 'p':
                chunks = self.chunk_text(child.text.strip('\n '))
                for chunk in chunks:
                    yield Document(
                        page_content=chunk,
                        metadata=dict(self.metadata(source=self.file_name, section=section, title=title)),
                    )

    def chunk_text(self, text) -> list[str]:
        """
        Due to the input text is a paragraph, to preserve the semantic meanings per chunk.
        Chunking the text more than 400 tokens, meanwhile, prevent generating small chunks less than 200 tokens.
        The range of tokens per chunk is 200 - 600.
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
            token_left = sum(
                token_per_sent[i:]
            )   # yes, I intend to keep this token in the result

            if accumulate_token <= 400:
                continue
            if token_left >= 200:
                chunked_texts.append(' '.join(sentences[next_start_i:i]))
                next_start_i = i
                accumulate_token = token
            else:
                chunked_texts.append(' '.join(sentences[next_start_i:]))
                break
        return chunked_texts


class FullTextLoader(Loader):
    """Full text loader, used for validation process (resolving abbreviation definitions) or as extracted object"""
    include_table = False
    max_token = 5000

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_name = file_path.split(os.sep)[-1]
        self.metadata = create_model('MetaData', source=(str, ...), title=(str, ...))
    
    def lazy_load(self) -> Iterator[Document]:
        """when I write this, I know I should write load function instead, but I want to keep the interface consistentency"""
        with open(self.file_path, encoding='utf8') as file:
            doc = file.read()
        soup = bs(doc, 'html.parser')
        if not self.include_table:
            tags = soup.find_all('table')
            if tags:
                for tag in tags:
                    tag.decompose()
        title = soup.find('title').text.strip('\n ')
        text = soup.get_text(separator='\n', strip=True)
        chunks = self.ensure_safe_len(text)
        for chunk in chunks:
            yield Document(page_content=chunk, metadata=dict(self.metadata(source=self.file_name, title=title)))

    async def alazy_load(self) -> AsyncIterator[Document]:
        async with aiofiles.open(self.file_path, encoding='utf8') as file:
            doc = await file.read()
        soup = bs(doc, 'html.parser')
        if not self.include_table:
            tags = soup.find_all('table')
            if tags:
                for tag in tags:
                    tag.decompose()
        title = soup.find('title').text.strip('\n ')
        text = soup.get_text(separator='\n', strip=True)
        chunks = self.ensure_safe_len(text)
        for chunk in chunks:
            yield Document(page_content=chunk, metadata=dict(self.metadata(source=self.file_name, title=title)))
    
    def ensure_safe_len(self, text):
        # TODO consolidate this to avoid of possibly context losing
        text_encoding = encoding.encode(text)
        start = 0
        chunk = encoding.decode(text_encoding[start: start + self.max_token])
        if chunk:
            yield chunk
            start += self.max_token
        else:
            return


# endregion
