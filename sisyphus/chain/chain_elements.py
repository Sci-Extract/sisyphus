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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Union, NamedTuple, Any

import tqdm
from pydantic import BaseModel, ValidationError
from openai import RateLimitError
from langchain_community.vectorstores import chroma
from langchain_community.callbacks import get_openai_callback
from langchain.output_parsers import PydanticToolsParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)

from sisyphus.patch import ChatOpenAIThrottle
from sisyphus.patch.throttle import chat_throttler, ChatThrottler
from sisyphus.chain.database import (
    DocDB,
    ResultDB,
    ExtractManager,
    add_manager_callback,
    aadd_manager_callback,
)
from sisyphus.chain.constants import *
from sisyphus.chain.paragraph import ParagraphExtend
from sisyphus.utils.run_bulk import bulk_runner


logger = logging.getLogger(__name__)

class DocInfo():
    def __init__(self, doc, info):
        self.doc = doc
        self.info = info

    def __repr__(self):
        return f"DocInfo(doc={repr(self.doc)}, info={repr(self.info)})"


def coercion_to_lambda(func):
    return ChainElementLambda(func)


class BaseElement(object):
    def __repr__(self):
        return self.__class__.__name__

    def invoke(self, input_):
        pass

    async def ainvoke(self, input_) -> Any:
        pass

    def __add__(self, other):
        if isinstance(other, Callable):
            other = coercion_to_lambda(other)
        return Chain(self, other)


class Filter(BaseElement):
    """Applied to Article level"""

    def __init__(
        self,
        db: Union[chroma.Chroma, DocDB],
        query: Optional[str] = None,
        filter_func: Optional[Callable[[Document], bool]] = None,
        with_abstract: bool = False,
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
        with_abstract : bool, optional
            contain abstract, not work for vector database. With this setting to True, you can access by abstract attr of the output docs, by default False
        """
        self.db = db
        self.query = query
        self.filter_func = filter_func
        self.with_abstract = with_abstract

    def locate(self, file_name) -> list[Document]:
        """select list of `Document` object in an article based on creterions, e.g., semantic similarity or use all.
        We assumed that articles are already been parsed and stored in database.
        """
        self.check_database()
        if self.query:
            docs = self.db.similarity_search(
                query=self.query, filter={'source': file_name}, k=10
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
                docs = self.db.get(source=file_name, with_abstract=self.with_abstract)
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
        filter_results = await asyncio.gather(
            *[self.afilter(doc) for doc in docs]
        )
        filter_docs = []
        for filter_boolean, doc in zip(filter_results, docs):
            if filter_boolean:
                filter_docs.append(doc)
        return filter_docs if filter_docs else None

    def invoke(self, input_):
        docs = self.locate(input_)
        filter_results = [self.filter_(doc) for doc in docs]
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
        self.cost = 0 # use langchain openai callback to track
        self.build_chain()

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
                f'prompt template arguments inconsistent with receive, expect: [examples, text], receive: {input_vars}'
            )
        self.chain = chain

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(min=2, max=10),
        after=_chat_throttler.retry_callback,
    )
    def _extract(self, doc: Document) -> Optional[list[BaseModel]]:
        """extract info from doc"""
        with get_openai_callback() as cb:
            result = self.chain.invoke(
                {'examples': self.examples, 'text': doc.page_content}
            )
            self.cost += cb.total_cost
        return result if result else None

    @retry(
        retry=retry_if_exception_type(ValidationError),
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(retry_times),
        retry_error_callback=lambda r: None,
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
        with get_openai_callback() as cb:
            result = await self.chain.ainvoke(
                {'examples': self.examples, 'text': doc.page_content}
            )
            self.cost += cb.total_cost
        return result if result else None

    @retry(
        retry=retry_if_exception_type(ValidationError),
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(retry_times),
        retry_error_callback=lambda r: None,
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
    
    def invoke(self, docs):
        results = [self.extract(doc) for doc in docs]
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
                to_validates = await asyncio.gather(
                    *[
                        asyncio.to_thread(gadget, to_validate)
                        for to_validate in to_validates
                    ]
                )
            else:
                # validates_copy = to_validates
                to_validates = await asyncio.gather(
                    *[gadget(to_validate) for to_validate in to_validates]
                )
                # for origin, derived in zip(validates_copy, to_validates):
                #     if derived is None:
                #         print(origin)
            to_validates = list(
                filter(None, to_validates)
            )   # remove None element
        return to_validates if to_validates else None

    async def ainvoke(
        self, to_validates: list[DocInfo]
    ) -> Optional[list[DocInfo]]:
        return await self.avalidate(to_validates)
    
    def invoke(self, to_validates):
        return self.validate(to_validates)


class Writer(BaseElement):
    """Write to sql base, applied to `Document` level"""

    def __init__(self, result_db: ResultDB):
        self.result_db = result_db
    
    def save(self, paragraph: ParagraphExtend):
        """save document and correspond results"""
        data = paragraph.data
        if data:
            self.result_db.save_result(
                text=paragraph.page_content,
                metadata=paragraph.metadata,
                results=data
            )

    async def asave(self, paragraphs):
        await asyncio.to_thread(self.save, paragraphs)

    async def ainvoke(self, paragraphs) -> None:
        await asyncio.gather(
            *[
                self.asave(paragraph)
                for paragraph in paragraphs
            ]
        )
    
    def invoke(self, paragraphs: list[ParagraphExtend]):
        for paragraph in paragraphs:
            self.save(paragraph)


class ChainElementLambda(BaseElement):
    """useful when you want to intercept the output of one of the basic chain elements. e.g.,
    - print to the terminal instead of saving to sql base"""

    def __init__(self, func: Callable[[Any], Optional[Any]]):
        self.func = func

    async def ainvoke(self, input_: Any) -> Optional[Any]:
        if not inspect.iscoroutinefunction(self.func):
            return await asyncio.to_thread(self.func, input_)
        return await self.func(input_)
    
    def invoke(self, input_):
        return self.func(input_)


class Chain:
    def __init__(self, *components: BaseElement):
        self.components = components

    def __add__(self, other):
        if isinstance(other, BaseElement):
            return Chain(*self.components, other)
        if isinstance(other, Callable):
            other = coercion_to_lambda(other)
            return Chain(*self.components, other)
        return Chain(*self.components, *other.components)

    async def acompose(self, input_):
        origin_input = input_
        for index, component in enumerate(self.components):
            input_ = await component.ainvoke(input_)
            if input_ == FAILED:
                return FAILED
            if not input_  and index < len(self.components) - 1:
                logger.debug('file: %s no result find', origin_input)
                return
        return input_
    
    def compose(self, input_):
        origin_input = input_
        for index, component in enumerate(self.components):
            input_ = component.invoke(input_)
            if input_ == FAILED:
                return FAILED
            if not input_ and index < len(self.components) - 1:
                logger.debug('file: %s no result find', origin_input)
                return
        return input_



async def asupervisor(chain: Chain, directory: str, batch_size: int):
    """Deprecated!"""
    file_name_full = glob.glob(os.path.join(directory, '*.html'))
    file_names = [name.split(os.sep)[-1] for name in file_name_full]

    for index in range(0, len(file_names), batch_size):
        coros = []
        for file_name in file_names[index : index + batch_size]:
            coros.append(chain.acompose(file_name))
        for coro in tqdm.tqdm(asyncio.as_completed(coros), total=len(coros)):
            await coro


async def run_chains_with_extraction_history(
    chain: Chain, directory: Optional[str], batch_size: int, namespace: str, extract_nums: Optional[int] = None, file_names: list[str] = None
):
    """run multiple chains asynchronously with extraction history.
    Args:
        namespace: Give a name to current task. For example 'nlo/band_gap' is a good name for extracting band gap from NLO papers.
    """
    if not file_names:
        file_name_full = glob.glob(os.path.join(directory, '*.html'))
        file_names = [name.split(os.sep)[-1] for name in file_name_full]
        if extract_nums:
            file_names = file_names[:extract_nums]

    # skip extracted ones
    manager = ExtractManager(
        namespace,
        db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME),
    )
    manager.create_schema()
    exists = manager.exists(file_names)
    file_names = [
        file_name for file_name, exist in zip(file_names, exists) if not exist
    ]
    logger.debug('total processed files: %d', len(file_names))
    if not file_names:
        raise ValueError('no file needed to be extracted')
    # register manager callback
    runnable = aadd_manager_callback(chain.acompose, manager)

    await bulk_runner(
        task_producer=file_names,
        repeat_times=None,
        batch_size=batch_size,
        runnable=runnable,
    )

def run_chains_with_extarction_history_multi_threads(
    chain: Chain, directory: Optional[str], batch_size: int, namespace: str, extract_nums: Optional[int] = None, given_names: list[str] = None
):
    """run multiple chains with extraction history in multi-threads"""
    file_names = given_names
    if not file_names:
        file_name_full = glob.glob(os.path.join(directory, '*.html'))
        file_names = [name.split(os.sep)[-1] for name in file_name_full]
        
    # skip extracted ones
    manager = ExtractManager(
        namespace,
        db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME),
    )
    manager.create_schema()
    exists = manager.exists(file_names)
    file_names = [
        file_name for file_name, exist in zip(file_names, exists) if not exist
    ]
    if not given_names and extract_nums:
        file_names = file_names[:extract_nums]
    logger.debug('total processed files: %d', len(file_names))
    if not file_names:
        raise ValueError('no file needed to be extracted')
    # register manager callback
    runnable = add_manager_callback(chain.compose, manager)

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(runnable, file_name) for file_name in file_names]
        for future in tqdm.tqdm(as_completed(futures), total=len(file_names)):
            future.result()

def run_chains_with_extraction_history_for_one(
    chain: Chain, file_name: str, namespace: str
):
    """run chain with extraction history for one file"""
    manager = ExtractManager(
        namespace,
        db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME),
    )
    manager.create_schema()
    exists = manager.exists([file_name])
    if exists[0]:
        raise ValueError('no file needed to be extracted')
    runnable = add_manager_callback(chain.compose, manager)
    runnable(file_name)
