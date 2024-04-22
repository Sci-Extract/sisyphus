# -*- coding:utf-8 -*-
'''
@File    :   vectordb.py
@Time    :   2024/04/22 17:12:57
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   create vector database
'''

import asyncio
import os
import glob
import logging
import logging.config

from tqdm import tqdm
from langchain.indexes import SQLRecordManager
from langchain_community.vectorstores import faiss

from .indexes import aindex
from .archive import ArticleLoader
from sisyphus.patch.embed_patch import OpenAIEmbeddingThrottle


logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')

embedding = OpenAIEmbeddingThrottle()
faiss_store = faiss.FAISS.from_texts(texts=[""], embedding=embedding) # instantiate faiss
collection_name = "index"
namespace = f"faiss/{collection_name}"
sql_path = os.path.join('db', 'record_manager.sql')
record_manager = SQLRecordManager(
    namespace, db_url="sqlite+aiosqlite:///" + sql_path, async_mode=True
)

async def aembed_doc(file_path):
    loader = ArticleLoader(file_path)
    docs = [doc async for doc in loader.alazy_load()]
    info = await aindex(
        docs_source=docs,
        record_manager=record_manager,
        vector_store=faiss_store,
        cleanup='incremental',
        source_id_key='source'
    )
    return info

async def supervisor(file_folder, batch_size: int = 10):
    """
    supervisor : manage the index process.

    Parameters
    ----------
    file_folder : str
        the folder contains html files parsed by chempp
    batch_size: int
        the parallel processing batch size
    """
    await record_manager.acreate_schema()
    file_paths = glob.glob(os.path.join(file_folder, '*.html'))
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i: i+batch_size]
        coros = [aembed_doc(file_path) for file_path in batch]
        iter_ = asyncio.as_completed(coros)
        if logger.level > 10: # not debug level
            iter_ = tqdm(iter_, total=len(batch))
        for coro in iter_:
            info = await coro
            logger.debug(info)
    faiss_store.save_local("db")

def index(file_folder, batch_size=10):
    """
    index method, runs only one time.
    """
    asyncio.run(supervisor(file_folder, batch_size))
