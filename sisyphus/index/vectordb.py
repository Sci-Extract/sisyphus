# -*- coding:utf-8 -*-
"""
@File    :   vectordb.py
@Time    :   2024/04/22 17:12:57
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   create vector database backend by chroma database. support both sync and async.
"""

import asyncio
import os
import glob
import logging
import logging.config

from tqdm import tqdm
import chromadb
from langchain.indexes import index, SQLRecordManager
from langchain_community.vectorstores import chroma

from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    AsyncChroma,
    aembed_httpx_client,
)
from .indexes import aindex
from .archive import ArticleLoader


logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')

embedding = OpenAIEmbeddingThrottle(async_client=aembed_httpx_client)


async def aembed_doc(file_path, record_manager, vector_store):
    loader = ArticleLoader(file_path)
    info = await aindex(
        docs_source=loader,
        record_manager=record_manager,
        vector_store=vector_store,
        cleanup='incremental',
        source_id_key='source',
    )
    return info


def embed_doc(file_path, record_manager, vector_store):
    loader = ArticleLoader(file_path)
    info = index(
        docs_source=loader,
        record_manager=record_manager,
        vector_store=vector_store,
        cleanup='incremental',
        source_id_key='source',
    )
    return info


async def asupervisor(
    client, file_folder, collection_name, batch_size: int = 10
):
    """
    asupervisor : manage the index process.

    Parameters
    ----------
    file_folder : str
        the folder contains html files parsed by chempp
    batch_size: int
        the parallel processing batch size
    """
    namespace = f'chroma/{collection_name}'
    sql_path = os.path.join('db', 'record_manager.sql')
    record_manager = SQLRecordManager(
        namespace, db_url='sqlite+aiosqlite:///' + sql_path, async_mode=True
    )
    await record_manager.acreate_schema()
    db = AsyncChroma(
        collection_name, client=client, embedding_function=embedding
    )
    file_paths = glob.glob(os.path.join(file_folder, '*.html'))
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i : i + batch_size]
        coros = [
            aembed_doc(file_path, record_manager, db) for file_path in batch
        ]
        iter_ = asyncio.as_completed(coros)
        if logger.level > 10:   # not debug level
            iter_ = tqdm(iter_, total=len(batch))
        for coro in iter_:
            info = await coro
            logger.debug(info)


def supervisor(client, file_folder, collection_name):
    """
    supervisor : manage the index process.

    Parameters
    ----------
    file_folder : str
        the folder contains html files parsed by chempp
    """
    namespace = f'chroma/{collection_name}'
    sql_path = os.path.join('db', 'record_manager.sql')
    record_manager = SQLRecordManager(
        namespace, db_url='sqlite:///' + sql_path
    )
    record_manager.create_schema()
    db = chroma.Chroma(
        collection_name, client=client, embedding_function=embedding
    )
    file_paths = glob.glob(os.path.join(file_folder, '*.html'))
    iter_ = iter(file_paths)
    if logger.level > 10:   # not debug level
        iter_ = tqdm(iter_, total=len(file_paths))
    for file_path in iter_:
        info = embed_doc(file_path, record_manager, db)
        logger.debug(info)


def acreate_vectordb(file_folder, collection_name, batch_size=10):
    """
    create vector database. Ensuring database consistency, means that one can run this process multiple times
    """
    client = chromadb.HttpClient()
    asyncio.run(asupervisor(client, file_folder, collection_name, batch_size))


def create_vectordb(file_folder, collection_name):
    """
    sync version
    """
    client = chromadb.HttpClient()
    supervisor(client, file_folder, collection_name)
