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
from typing import Optional, Callable, Union, NamedTuple, Any

import tqdm
from openai import RateLimitError
from langchain_community.vectorstores import chroma
from langchain.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

from sisyphus.patch import ChatOpenAIThrottle
from sisyphus.patch.throttle import chat_throttler, ChatThrottler
from sisyphus.chain.database import DocDB, ResultDB, ExtractManager, add_manager_callback
from sisyphus.utils.run_bulk import bulk_runner


logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')
RECORD_LOCATION = 'record'
RECORD_NAME = 'extract_record.sqlite'


class DocInfo(NamedTuple):
    doc: Document
    info: list[BaseModel]


class BaseElement(object):
    def __repr__(self):
        return self.__class__.__name__
    
    def invoke(self):
        pass

    async def ainvoke(self) -> Any:
        pass

    def __add__(self, other):
        return Chain(self, other)


class Filter(BaseElement):
    """Applied to Article level"""

    def __init__(
        self,
        db: Union[chroma.Chroma, DocDB],
        query: Optional[str] = None,
        filter_func: Optional[Callable[[], bool]] = None,
    ):
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
                docs = self.db.get(source=file_name)
            else:
                raise NotImplementedError('use chroma or DocDB!')
        return docs

    async def alocate(self, file_name) -> list[Document]:
        """async version, overwrite this if has native async method"""
        return await asyncio.to_thread(self.locate, file_name)

    def filter_(self, doc: Document) -> bool:
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
        return await asyncio.to_thread(self.filter_, doc)
    
    
    async def ainvoke(self, input_) -> Optional[list[Document]]:
        docs = await self.alocate(input_)
        filter_results = await asyncio.gather(*[self.afilter(doc) for doc in docs])
        filter_docs = []
        for filter_boolean, doc in zip(filter_results, docs):
            if filter_boolean:
                filter_docs.append(doc)
        return filter_docs if filter_docs else None
                

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
                    f'{type(self.db)} type database does not support semantic filter, set query to None '
                    'or use Chroma database'
                )


DEFAULT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a helpful assistant to extract information from text, if you are do not know the value of an asked field, return null for the value. Call the function multiple times if needed',
        ),
        MessagesPlaceholder(variable_name='examples'),
        ('human', '{text}'),
    ]
)


class Extractor(BaseElement):
    """Applied to `Document` object, use openai tool call"""

    _chat_throttler: ChatThrottler = chat_throttler
    """used for cool down process while hit 429 error"""
    retry_times: int = 2
    """default retry times"""

    def __init__(
        self,
        chat_model: ChatOpenAIThrottle,
        pydantic_model: BaseModel,
        examples: list[BaseMessage] = list(),
    ):
        self.chat_model = chat_model
        self.pydantic_models = [pydantic_model]
        self.examples = examples if examples else []
        self.prompt = DEFAULT_PROMPT_TEMPLATE
        self.parser = PydanticToolsParser(tools=self.pydantic_models)
        self.chain = self.build_chain()

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
    def _extract(self, doc: Document) -> Optional[list[BaseModel]]:
        """extract info from doc"""
        result = self.chain.invoke(
            {'examples': self.examples, 'text': doc.page_content}
        )
        return result if result else None

    @retry(
        retry=retry_if_exception_type(ValidationError),
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(retry_times),
        retry_error_callback=lambda r: None
    )
    def extract(self, doc: Document) -> Optional[list[BaseModel]]:
        """retry extract if pydantic validation error, default to `retry_times`"""
        result = self._extract(doc)
        return result

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(min=2, max=10),
        after=_chat_throttler.retry_callback,
    )
    async def _aextract(self, doc: Document) -> Optional[list[BaseModel]]:
        """async version for extracting info from doc"""
        result = await self.chain.ainvoke(
            {'examples': self.examples, 'text': doc.page_content}
        )
        return result if result else None
    
    @retry(
        retry=retry_if_exception_type(ValidationError),
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(retry_times),
        retry_error_callback=lambda r: None
    )
    async def aextract(self, doc: Document) -> Optional[list[BaseModel]]:
        """retry extract if pydantic validation error, default to `retry_times`"""
        result = await self._aextract(doc)
        return result
    
    async def ainvoke(self, docs: list[Document]) -> Optional[list[DocInfo]]:
        results = await asyncio.gather(*[self.aextract(doc) for doc in docs])
        docinfos = []
        for result, doc in zip(results, docs):
            if result:
                docinfos.append(DocInfo(doc, result))
        return docinfos if docinfos else None


T = Callable[[DocInfo], Optional[DocInfo]]

class Validator(BaseElement):
    """Applied to article level, refer to sisyphus/chain/validators for more information"""
    # TODO: use chemdataextrator to resolve chemical name abbreviations, but I only need the abbreviation detection algorithm.

    def __init__(self):
        self._gadget: list[T] = []

    def add_gadget(self, gadget: T):
        """add validate function to validator, note that function receives `DocInfo` object"""
        self._gadget.append(gadget)
    
    def validate(self, to_validates: list[DocInfo]):
        before_processed = to_validates
        after_processed = []
        for gadget in self._gadget:
            for to_validate in before_processed:
                after_processed.append(gadget(to_validate))
            before_processed = list(filter(None, after_processed))
            after_processed = []
        return before_processed if before_processed else None

    async def avalidate(self, to_validates: list[DocInfo]):
        for gadget in self._gadget:
            if not inspect.iscoroutinefunction(gadget):
                to_validates = await asyncio.gather(*[asyncio.to_thread(gadget, to_validate) for to_validate in to_validates])
            else:
                # validates_copy = to_validates
                to_validates = await asyncio.gather(*[gadget(to_validate) for to_validate in to_validates])
                # for origin, derived in zip(validates_copy, to_validates):
                #     if derived is None:
                #         print(origin)
            to_validates = list(filter(None, to_validates)) # remove None element
        return to_validates if to_validates else None
    
    async def ainvoke(self, to_validates: list[DocInfo]) -> Optional[list[DocInfo]]:
        return await self.avalidate(to_validates)
    
    
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

    async def asave(self, results, document):
        await asyncio.to_thread(self.save, results, document)
    
    async def ainvoke(self, doc_with_info: list[DocInfo]) -> None:
        await asyncio.gather(
            *[self.asave(docinfo.info, docinfo.doc) for docinfo in doc_with_info]
        )


class Chain:
    def __init__(self, *components: BaseElement):
        self.components = components

    def __add__(self, other):
        if isinstance(other, BaseElement):
            return Chain(*self.components, other)
        return Chain(*self.components, *other.components)
    
    async def acompose(self, input_):
        origin_input = input_
        for index, component in enumerate(self.components):
            input_ = await component.ainvoke(input_)
            if input_ is None and index < len(self.components) - 1:
                logger.debug("file: %s no result find", origin_input)
                return
        return input_


async def asupervisor(chain: Chain, directory: str, batch_size: int):
    """Deperacated!"""
    file_name_full = glob.glob(os.path.join(directory, '*.html'))
    file_names = [name.split(os.sep)[-1] for name in file_name_full]

    for index in range(0, len(file_names), batch_size):
        coros = []
        for file_name in file_names[index : index + batch_size]:
            coros.append(chain.acompose(file_name))
        for coro in tqdm.tqdm(asyncio.as_completed(coros), total=len(coros)):
            await coro


async def run_chains_with_extraction_history(chain: Chain, directory: str, batch_size: int, namespace: str):
    """run multiple chains asynchronously with extraction history. 
    - provide different namespace for different extraction tasks, better use the name of your extraction object, e.g., nlo/band_gap"""
    file_name_full = glob.glob(os.path.join(directory, '*.html'))
    file_names = [name.split(os.sep)[-1] for name in file_name_full]

    # skip extracted ones
    manager = ExtractManager(namespace, db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME))
    manager.create_schema()
    exists = manager.exists(file_names)
    file_names = [file_name for file_name, exist in zip(file_names, exists) if not exist]
    logger.debug('total processed files: %d', len(file_names))
    if not file_names:
        raise ValueError('no file needed to be extracted')
    # register manager callback
    runnable = add_manager_callback(chain.acompose, manager)

    await bulk_runner(
        task_producer=file_names,
        repeat_times=None,
        batch_size=batch_size,
        runnable=runnable
    )
