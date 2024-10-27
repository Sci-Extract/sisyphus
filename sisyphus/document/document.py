# -*- coding:utf-8 -*-
"""
@File    :   document.py
@Time    :   2024/05/17 11:53:31
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Used for testing, web API. Use in memory sqlite db instead of persisted db
"""

import asyncio
import os
from typing import NamedTuple

from langchain.indexes import SQLRecordManager
from langchain_core.documents import Document as langchain_document
from pydantic import BaseModel
from sqlmodel import create_engine

from sisyphus.index.indexing import aembed_doc, save_doc
from sisyphus.index.loader import Loader, ArticleLoader
from sisyphus.index.langchain_index import aindex
from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    ChatOpenAIThrottle,
    aembed_httpx_client,
    achat_httpx_client,
    AsyncChroma,
)
from sisyphus.chain.database import DocDB
from sisyphus.chain import Chain, Filter, Extractor, Validator, Writer


embed = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
chat = ChatOpenAIThrottle(http_async_client=achat_httpx_client)
record_manager = SQLRecordManager(
    'database/record', db_url='sqlite+aiosqlite://', async_mode=True
)
asyncio.run(record_manager.acreate_schema())
vector_store = AsyncChroma('chroma', embed)
plain_store = DocDB(create_engine('sqlite://'))


class UserParams(NamedTuple):
    full_text: bool
    enabling_semantic_search: bool
    query: str
    pydantic_model: BaseModel


class Document:
    """sisyphuse document"""

    def __init__(self, file_path: str, user_params: UserParams):
        self.file_path = file_path
        self.user_params = user_params

    async def aindexing(self):
        if self.user_params.enabling_semantic_search:
            await aembed_doc(
                self.file_path,
                record_manager,
                vector_store,
                self.user_params.full_text,
            )
        else:
            await asyncio.to_thread(
                save_doc,
                self.file_path,
                plain_store,
                self.user_params.full_text,
            )

    async def aextract_with_chain(self, chain: Chain):
        file_name = self.file_path.split(os.sep)[-1]
        await chain.acompose(file_name)


class DocWriter(Writer):
    def __init__(self):
        pass

    def save(self, results: list[BaseModel], document: langchain_document):
        print(f'Document: {document.metadata["source"]}\n')
        for result in results:
            print(result)


def create_chain_by_params(userparams: UserParams):
    if userparams.enabling_semantic_search:
        filter_ = Filter(vector_store, userparams.query)
    else:
        filter_ = Filter(plain_store)
    extractor = Extractor(chat, [userparams.pydantic_model])
    validator = Validator()
    writer = DocWriter()
    chain = Chain(filter_, extractor, validator, writer)
    return chain
