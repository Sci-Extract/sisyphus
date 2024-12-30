"""This module provide some easy configurable chain elements like `Extractor`, `Validator`"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from .chain_elements import DocInfo, Document
from ..utils.helper_functions import return_valid


def customized_extractor(my_extractor: Callable[[Document], Optional[list[BaseModel]]], mode: Literal['async', 'thread', 'normal'], concurrent_number: int = 5) -> Callable:
    """
    customized extractor

    Args:
        my_extractor (Callable[[Document], list[BaseModel]]): custom extractor function
        mode (Literal['async', 'thread', 'normal']): async, thread or just run one by one
        concurrent_number (int, optional): maximum concurrently running tasks. Defaults to 5.

    Returns:
        Callable: a callable called by the chain, do not call it by yourself.
    """
    @return_valid
    def do_extract_multi_threads(docs: list[Document]) -> list[DocInfo]:
        with ThreadPoolExecutor(max_workers=concurrent_number) as executor:
            results = executor.map(my_extractor, docs)
        zipped_results = filter(lambda x: x[1], zip(docs, results))
        doc_infos = [DocInfo(doc=doc, info=result) for doc, result in zipped_results]
        return doc_infos
    
    @return_valid
    async def do_extract_async(docs: list[Document]) -> list[DocInfo]:
        def wrapped_with_sema(sema):
            async def wrapped(doc):
                async with sema:
                    await my_extractor(doc)
            return wrapped
        sema = asyncio.Semaphore(concurrent_number)
        tasks = [wrapped_with_sema(sema)(doc) for doc in docs]
        results = await asyncio.gather(*tasks)
        zipped_results = filter(lambda x: x[1], zip(docs, results))
        doc_infos = [DocInfo(doc=doc, info=result) for doc, result in zipped_results]
        return doc_infos

    @return_valid
    def do_extract_normal(docs: list[Document]) -> list[DocInfo]:
        results = [my_extractor(doc) for doc in docs]
        zipped_results = filter(lambda x: x[1], zip(docs, results))
        doc_infos = [DocInfo(doc=doc, info=result) for doc, result in zipped_results]
        return doc_infos

    if mode == 'async':
        return do_extract_async
    if mode == 'thread':
        return do_extract_multi_threads
    if mode == 'normal':
        return do_extract_normal
    raise ValueError('mode should be one of "async", "thread" or "normal"')

def customized_validator(my_validator: Callable[[DocInfo], Optional[DocInfo]], mode: Literal['async', 'thread', 'normal'], concurrent_number: int = 5) -> Callable:
    """
    customized validator

    Args:
        my_validator (Callable[[DocInfo], Optional[DocInfo]]): custom validator function
        mode (Literal['async', 'thread', 'normal']): async, thread or just run one by one
        concurrent_number (int, optional): maximum concurrently running tasks. Defaults to 5.

    Returns:
        Callable: a callable called by the chain, do not call it by yourself.
    """
    @return_valid
    def do_validate_multi_threads(docinfos: list[DocInfo]) -> list[DocInfo]:
        with ThreadPoolExecutor(max_workers=concurrent_number) as executor:
            results = executor.map(my_validator, docinfos)
        results = list(filter(None, results))
        return results
    
    @return_valid
    async def do_validate_async(docs: list[DocInfo]) -> list[DocInfo]:
        def wrapped_with_sema(sema):
            async def wrapped(docinfo):
                async with sema:
                    await my_validator(docinfo)
            return wrapped
        sema = asyncio.Semaphore(concurrent_number)
        tasks = [wrapped_with_sema(sema)(doc) for doc in docs]
        results = await asyncio.gather(*tasks)
        results = filter(None, results)
        return list(results)

    @return_valid
    def do_validate_normal(docs: list[DocInfo]) -> list[DocInfo]:
        results = [my_validator(doc) for doc in docs]
        results = filter(None, results)
        return list(results)

    if mode == 'async':
        return do_validate_async
    if mode == 'thread':
        return do_validate_multi_threads
    if mode == 'normal':
        return do_validate_normal
    raise ValueError('mode should be one of "async", "thread" or "normal"')
