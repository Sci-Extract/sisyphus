# -*- coding:utf-8 -*-
"""
@File    :   chain_elements.py
@Time    :   2024/04/29 20:06:00
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Defnition of Filter, Extractor, Validator, SqlWriter
"""

import asyncio
import os
import glob
import logging
import inspect
import logging.config
from pydoc import doc
from typing import Optional, Callable, Union
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
from tenacity import retry, retry_if_exception_type, wait_exponential
from sqlmodel import Session

from sisyphus.patch import ChatOpenAIThrottle
from sisyphus.patch.throttle import chat_throttler, ChatThrottler
from sisyphus.chain.database import DocBase, ResultBase, DocDB, ResultDB
from sisyphus.utils.run_bulk import bulk_runner

logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')


class BaseElement(object):
    def __repr__(self):
        return self.__class__.__name__


class Filter(BaseElement):
    """Applied to Article level"""

    def __init__(
        self,
        db: Union[chroma.Chroma, DocDB],
        query: Optional[str] = None,
        filter_func: Optional[Callable] = None,
    ):   # TODO: db parameter support plain database using sqlite
        """
        create filter

        Parameters
        ----------
        db : Union[chroma.Chroma, DocDB]
            use chroma to activate query parameter, use DocDB for plain filter
        query : Optional[str], optional
            used for semantic search, by default None
        filter_func : Optional[Callable], optional
            other customize function, by default None
        """
        self.db = db
        self.query = query
        self.filter_func = filter_func

    def locate(self, file_name) -> list[Document]:
        """select list of `Document` object in an article based on creterions, e.g., semantic similarity or use all.
        We assumed that articles are already been parsed and stored in database.
        """
        self.check_database()
        if self.query:
            docs = self.db.similarity_search(
                query=self.query, filter={'source': file_name}
            )
        else:
            if isinstance(self.db, chroma.Chroma):
                results = self.db._collection.get(
                    where=dict(source=file_name),
                    include=['documents', 'metadatas'],
                )
                docs = self._convert_chroma_result_to_document(results)
            elif isinstance(self.db, DocDB):
                self.db: DocDB
                results = self.db.get(source=file_name)
            else:
                raise NotImplementedError('use chroma or DocDB!')
        return docs

    async def alocate(self, file_name) -> list[Document]:
        """async version, overwrite this if has native async method"""
        return await asyncio.to_thread(self.locate, file_name)

    def filter(self, doc: Document) -> bool:
        """filter on document, return boolean"""
        if not self.filter_func:
            return True   # when func not provided
        paras = list(inspect.signature(self.filter_func).parameters.keys())
        if len(paras) != 1:
            raise ValueError('filter function only takes one argument!')
        res = self.filter_func(doc)
        if not isinstance(res, bool):
            raise ValueError('filter function result should be a boolean')
        return res

    async def afilter(self, doc: Document) -> bool:
        return await asyncio.to_thread(self.filter, doc)

    def _convert_chroma_result_to_document(self, results: dict[str, list]):
        return [
            Document(page_content=document, metedata=metadata)
            for document, metadata in zip(
                results['documnets'], results['metadatas']
            )
        ]

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

    _chat_throttler: ChatThrottler = chat_throttler
    """used for cool down process while hit 429 error"""

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
        wait=wait_exponential(min=2, max=10),
        after=_chat_throttler.retry_callback,
    )
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
        wait=wait_exponential(min=2, max=10),
        after=_chat_throttler.retry_callback,
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

    def __init__(self, engine, doc_orm: DocBase, result_orm: ResultBase):
        self.engine = engine
        self.doc_orm = doc_orm
        self.result_orm = result_orm

    def save(self, results: list[BaseModel], document: Document):
        with Session(bind=self.engine) as session, session.begin():
            doc_sqlmodel = self._convert_to_doc_sqlmodel(document)
            for result in results:
                params = dict(result)
                result_sqlmodel = self.result_orm(**params)
                doc_sqlmodel.results.append(result_sqlmodel)
            session.add(doc_sqlmodel)

    def asave(self, *args, **kwargs):
        raise NotImplementedError()

    def _convert_to_doc_sqlmodel(self, doc: Document):
        return self.doc_orm(
            page_content=doc.page_content,
            source=doc.metadata['source'],
            section=doc.metadata['section'],
            title=doc.metadata['title'],
        )

class Writer(BaseElement):
    """Write to sql base, applied to `Document` level"""

    def __init__(self, result_db: ResultDB):
        self.result_db = result_db

    def save(self, results: list[BaseModel], document: Document):
        """save document and correspond results"""
        self.result_db.save_result(
            text=document.page_content,
            metadata=document.metadata,
            results=results
        )

    def asave(self):
        raise NotImplementedError

class Chain(BaseElement):
    """Chain through all the elements"""

    def __init__(
        self,
        filter: Filter,
        extractor: Extractor,
        validator: Validator,
        writer: Writer,
    ):
        self.filter = filter
        self.extractor = extractor
        self.validator = validator
        self.writer = writer

    def compose(self, file_name):
        """Extract aritlce based on extraction base elements"""
        docs = self.filter.locate(file_name)
        filter_docs = [doc for doc in docs if self.filter.filter(doc)]
        doc_info: list[DocInfo]
        doc_info = []
        for doc in filter_docs:
            results = self.extractor.extract(doc)
            if results:
                doc_info.append(DocInfo(doc, results))
        validate_doc_info = self.validator.validate(doc_info)
        for doc, info in validate_doc_info:
            self.writer.save(results=info, document=doc)

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

        coros = [extract_on_docuemnt(doc) for doc in docs]
        await asyncio.gather(*coros)
        validate_doc_info = await self.validator.avalidate(doc_info)
        # presently using sync save
        for doc, info in validate_doc_info:
            self.writer.save(results=info, document=doc)
        
        logger.info('file: %s extract process finished')


async def asupervisor(chain: Chain, directory: str, batch_size: int):
    """Asynchronously run `Chain` object"""
    file_name_full = glob.glob(os.path.join(directory, '*.html'))
    file_names = [name.split(os.sep)[-1] for name in file_name_full]

    for index in range(0, len(file_names), batch_size):
        coros = []
        for file_name in file_names[index : index + batch_size]:
            coros.append(chain.acompose(file_name))
        for coro in tqdm.tqdm(asyncio.as_completed(coros), total=len(coros)):
            await coro

async def run_chains(chain: Chain, directory: str, batch_size: int):
    file_name_full = glob.glob(os.path.join(directory, '*.html'))
    file_names = [name.split(os.sep)[-1] for name in file_name_full]

    await bulk_runner(
        task_producer=file_names,
        repeat_times=None,
        batch_size=batch_size,
        runnable=chain.acompose
    )
