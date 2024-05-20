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
import functools
import logging
import logging.config

from tqdm import tqdm
import chromadb
from langchain.indexes import index, SQLRecordManager
from langchain_community.vectorstores import chroma

from sqlmodel import create_engine
from sisyphus.chain.database import DocDB
from sisyphus.utils.run_bulk import bulk_runner
from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    AsyncChroma,
    aembed_httpx_client,
)
from .langchain_index import aindex
from .loader import ArticleLoader, Loader


DEFAULT_DB_DIR = 'db'
logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')

embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)


def choose_loader(file_path, full_text: bool) -> Loader:
    # TODO: based on file name
    if full_text: # return full text loader
        pass
    return ArticleLoader(file_path)

async def aembed_doc(file_path, record_manager, vector_store, full_text: bool = False):
    loader = choose_loader(file_path, full_text)
    info = await aindex(
        docs_source=loader,
        record_manager=record_manager,
        vector_store=vector_store,
        cleanup='incremental',
        source_id_key='source',
    )
    return info


def embed_doc(file_path, record_manager, vector_store, full_text: bool = False):
    loader = choose_loader(file_path, full_text)
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
    sql_path = os.path.join('db', 'record_manager.sqlite')
    record_manager = SQLRecordManager(
        namespace, db_url='sqlite+aiosqlite:///' + sql_path, async_mode=True
    )
    await record_manager.acreate_schema()
    db = AsyncChroma(
        collection_name, client=client, embedding_function=embedding
    )
    file_paths = glob.glob(os.path.join(file_folder, '*.html'))

    embed_runner = functools.partial(aembed_doc, record_manager=record_manager, vector_store=db)
    await bulk_runner(
        task_producer=file_paths,
        repeat_times=None,
        batch_size=batch_size,
        runnable=embed_runner
    )


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


def save_doc(file_path, database: DocDB, full_text: bool = False):
    # TODO: I should match use case to instantiate loader according to different publishers
    loader = choose_loader(file_path, full_text)
    documents = list(loader.lazy_load())
    texts = [document.page_content for document in documents]
    metadatas = [document.metadata for document in documents]
    database.save_texts(texts, metadatas)


def create_plaindb(file_folder, db_name):
    """
    create_plaindb: create database without the vector embeddings.

    Args:
        file_folder (str): the folder where to store articles
        db_name (str): the name of the database
    """
    sql_path = os.path.join(DEFAULT_DB_DIR, db_name)
    engine = create_engine('sqlite:///' + sql_path)
    db = DocDB(engine)
    db.create_db()

    file_paths = glob.glob(os.path.join(file_folder, '*.html'))
    for file_path in file_paths:
        save_doc(file_path, db)
