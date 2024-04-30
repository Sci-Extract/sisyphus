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
from sqlalchemy import create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session
)

encoding = tiktoken.get_encoding('cl100k_base')
MetaData = namedtuple("MetaData", "source section title")

# region loader
class ArticleLoader(BaseLoader):
    """convert article.html to langchain `Document` object"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = file_path.split(os.sep)[-1]

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

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding='utf8') as file:
            doc = file.read()
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
        Get rest sections.
        TODO: better use nested section name, e.g. Experimental/<sub_section> 
        """
        section = "abstract" # a few papers have abstract which more than one para
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
# endregion

# region build sql
class Base(DeclarativeBase):
    pass

class Doc(Base):
    __tablename__ = 'document'
    id: Mapped[int] = mapped_column(primary_key=True)
    page_content: Mapped[str] = mapped_column()
    source: Mapped[str] = mapped_column()
    section: Mapped[str] = mapped_column()
    title: Mapped[str] = mapped_column()

    def __repr__(self):
        return f'Doc(page_content={self.page_content[:20]}, source={self.source!r}, section={self.section!r}, title={self.title!r}'

def create_article_sqlite(file_folder, sql_name='article.sqlite', batch_size=10):
    """create article sqlite database"""
    engine_path = os.path.join('db', sql_name)
    engine = create_engine('sqlite:///' + engine_path)
    session = Session(bind=engine)
    Base.metadata.create_all(bind=engine)

    def convert_to_orm_doc(doc: Document):
        return Doc(
            page_content = doc.page_content,
            source = doc.metadata['source'],
            section = doc.metadata['section'],
            title = doc.metadata['title']
        )
    files = glob.glob(os.path.join(file_folder, '*.html'))

    session.begin()
    for file_path in files:
        loader = ArticleLoader(file_path)
        try:
            for doc in loader.lazy_load():
                orm_doc = convert_to_orm_doc(doc)
                session.add(orm_doc)
                
                if len(session.new) % batch_size == 0:
                    session.commit()
                    session.begin()  # Start a new transaction
                
            session.commit()
        except:
            # Rollback the transaction if an error occurs
            session.rollback()
            raise
        finally:
            # Close the session
            session.close()
