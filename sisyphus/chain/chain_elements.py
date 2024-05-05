# -*- coding:utf-8 -*-
"""
@File    :   chain_elements.py
@Time    :   2024/04/29 20:06:00
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Defnition of Parser, Filter, Extractor, Validator, SqlWriter
"""

import asyncio
import os
import glob
import logging
import inspect
import logging.config
from pydoc import doc
from typing import Optional, Callable
from collections import namedtuple

from openai import RateLimitError
import tqdm
from langchain_community.vectorstores import chroma
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.documents import Document
from langchain.output_parsers import PydanticToolsParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from sqlalchemy import create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    mapped_column,
    Mapped,
    relationship,
    Session,
)
from sqlalchemy import ForeignKey

from sisyphus.patch import ChatOpenAIThrottle


logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')


class BaseElement(object):
    def __repr__(self):
        return self.__class__.__name__


class Filter(BaseElement):
    """Applied to Article level"""

    def __init__(
        self,
        db: chroma.Chroma,
        query: Optional[str] = None,
        filter_func: Optional[Callable] = None,
    ):   # TODO: db parameter support plain database using sqlite
        self.db = db
        self.query = query
        self.filter_func = filter_func

    def locate(self, file_name) -> list[Document]:
        """select list of `Document` object in an article based on creterions, e.g., semantic similarity or use all"""
        self.check_database()
        if self.query:
            docs = self.db.similarity_search(
                query=self.query, filter={'source': file_name}
            )
        else:
            docs = self.db._collection.get(where=file_name)
            # convert raw docs to `Document`
            raise NotImplementedError(
                'not implement direct filter on doc right now!'
            )
        return docs

    async def alocate(self, file_name) -> list[Document]:
        """async version, overwrite this if has native async method"""
        return await asyncio.to_thread(self.locate, file_name)

    def filter(self, doc: Document) -> bool:
        """filter on document, return boolean"""
        if not self.filter_func:
            return True # when func not provided
        paras = list(inspect.signature(self.filter_func).parameters.keys())
        if len(paras) != 1:
            raise ValueError('filter function only takes one argument!')
        res = self.filter_func(doc)
        if not isinstance(res, bool):
            raise ValueError('filter function result should be a boolean')
        return res

    async def afilter(self, doc: Document) -> bool:
        return await asyncio.to_thread(self.filter, doc)

    def check_database(self) -> bool:
        """raise if query is given but without using chroma as database"""
        if self.query:
            if not isinstance(self.db, chroma.Chroma):
                raise ValueError(
                    f'{type(self.db)} type database does not support semantic filter, set query to None'
                    'or use Chroma database'
                )


DEFAULT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a helpful assistant to extract information from text, if you are do not know the vlaue of an asked field, return null for the value. Call the function multiple times if needed',
        ),
        MessagesPlaceholder(variable_name='examples'),
        ('human', '{text}'),
    ]
)


class Extractor(BaseElement):
    """Applied to `Document` object, use openai tool call"""

    def __init__(
        self,
        chat_model: ChatOpenAIThrottle,
        pydantic_models: list[BaseModel],
        examples: Optional[list[BaseMessage]] = None,
    ):
        self.chat_model = chat_model
        self.pydantic_models = pydantic_models
        self.examples = examples if examples else []
        self.prompt = DEFAULT_PROMPT_TEMPLATE
        self.pydantic_models = pydantic_models
        self.parser = PydanticToolsParser(tools=self.pydantic_models)

    def build_chain(self):
        chain = (
            self.prompt
            | self.chat_model.bind_tools(
                tools=self.pydantic_models, tool_choice='auto'
            )
            | self.parser
        )
        input_vars = self.prompt.input_variables
        if set(input_vars) != set(['examples', 'text']):
            raise ValueError(
                f'prompt template arguments inconsistent with receive, receive: [examples, text], expect: {input_vars}'
            )
        return chain

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(2),
    )    # TODO: add a callback function to notice stop the request for awhile
    def extract(self, doc: Document) -> Optional[list[BaseModel]]:
        """extract info from doc"""
        chain = self.build_chain()
        result = None
        try:
            result = chain.invoke(
                {'examples': self.examples, 'text': doc.page_content}
            )
        except ValidationError as e:
            logger.debug(e)
        return result

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(2),
    )
    async def aextract(self, doc: Document) -> Optional[list[BaseModel]]:
        """async version for extracting info from doc"""
        chain = self.build_chain()
        result = None
        try:
            result = await chain.ainvoke(
                {'examples': self.examples, 'text': doc.page_content}
            )
        except ValidationError as e:
            logger.debug(e)
        return result


DocInfo = namedtuple('DocInfo', 'doc info')
class Validator(BaseElement):
    """Applied to article level"""

    def __init__(self):
        pass

    def validate(self, to_validate: list[DocInfo]):
        # TODO resolve demonstrative pronouns
        return to_validate

    async def avalidate(self, to_validate: list[DocInfo]):
        return await asyncio.to_thread(self.validate, to_validate)


class SqlWriter(BaseElement):
    """Store pydantic model to sqlite, applied on `Document` level"""

    def __init__(
        self, db_name: str, pydantic_model: BaseModel
    ):
        self.pydantic_model = pydantic_model
        self.db_name = db_name
        self.engine = None

    class Base(DeclarativeBase):
        pass

    class Doc(Base):
        __tablename__ = 'document'
        id: Mapped[int] = mapped_column(primary_key=True)
        page_content: Mapped[str] = mapped_column()
        source: Mapped[str] = mapped_column()
        section: Mapped[str] = mapped_column()
        title: Mapped[str] = mapped_column()
        results: Mapped[list['Result']] = relationship(back_populates='document')

    class Result(Base):
        __tablename__ = 'result'
        id: Mapped[int] = mapped_column(primary_key=True)
        doc_id: Mapped[int] = mapped_column(ForeignKey('document.id'))
        document: Mapped['Doc'] = relationship(back_populates='results')
        # TODO: refactor needed !
        mof_name: Mapped[str] = mapped_column()
        gas_type: Mapped[str] = mapped_column()
        uptake: Mapped[str] = mapped_column()
        temperature: Mapped[Optional[str]] = mapped_column()
        pressure: Mapped[Optional[str]] = mapped_column()

    def _create(self):
        # TODO: find better way to convet pydantic to sqlalchemy orm object
        db_path = os.path.join('db', self.db_name)
        engine = create_engine('sqlite:///' + db_path)
        self.Base.metadata.create_all(engine)
        return engine

    def bootstrap(self):
        self.engine = self._create()
    
    def _convert_to_doc_orm(self, doc: Document):
        return self.Doc(
            page_content = doc.page_content,
            source = doc.metadata['source'],
            section = doc.metadata['section'],
            title = doc.metadata['title']
        )

    def save(self, results: list[BaseModel], document: Document):
        with Session(bind=self.engine) as session, session.begin():
            doc_orm = self._convert_to_doc_orm(document)
            for result in results:
                params = dict(result)
                result_orm = self.Result(**params)
                doc_orm.results.append(result_orm)
            session.add(doc_orm)

    def asave(self, results: list[BaseModel], document: Document):
        """uncertain about thread safe"""
        # TODO: implement async version of save
        raise NotImplementedError()

class Chain(BaseElement):
    """Chain through all the elements"""
    def __init__(
            self,
            filter: Filter,
            extractor: Extractor,
            validator: Validator,
            writer: SqlWriter
    ):
        self.filter = filter
        self.extractor = extractor
        self.validator = validator
        self.writer = writer
        self.writer.bootstrap()

    def compose(self, file_name):
        """Extract aritlce based on extraction base elements"""
        docs = self.filter.locate(file_name)
        filter_docs = [
            doc for doc in docs
            if self.filter.filter(doc)
        ]
        doc_info: list[DocInfo]
        doc_info = []
        for doc in filter_docs:
            results = self.extractor.extract(doc)
            if results:
                doc_info.append(DocInfo(doc, results))
        validate_doc_info = self.validator.validate(doc_info)
        for doc, info in validate_doc_info:
            self.writer.save(
                results=info,
                document=doc
            )
    
    async def acompose(self, file_name):
        """Async version"""
        docs = await self.filter.alocate(file_name)
        doc_info: list[DocInfo]
        doc_info = []
        async def extract_on_docuemnt(doc):
            sentinel = await self.filter.afilter(doc)
            if not sentinel:
                return
            results = await self.extractor.aextract(doc)
            if results:
                doc_info.append(DocInfo(doc, results))
        coros = [
            extract_on_docuemnt(doc) for doc in docs
        ]
        await asyncio.gather(*coros)
        validate_doc_info = await self.validator.avalidate(doc_info)
        # presently using sync save
        for doc, info in validate_doc_info:
            self.writer.save(
                results=info,
                document=doc
            )


async def asupervisor(chain: Chain, directory: str, batch_size: int):
    """Asynchronously run `Chain` object"""
    file_name_full = glob.glob(os.path.join(directory, '*.html'))
    file_names = [name.split(os.sep)[-1] for name in file_name_full]    
    
    for index in range(0, len(file_names), batch_size):
        coros = []
        for file_name in file_names[index: index + batch_size]:
            coros.append(chain.acompose(file_name))
        for coro in tqdm.tqdm(asyncio.as_completed(coros), total=len(coros)):
            await coro 
