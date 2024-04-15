import os
from typing import Tuple

import chromadb
import httpx
from numpy import array
from openai import AsyncOpenAI
from pydantic import BaseModel

from .parallel_processor import EmbeddingRequest, CompletionRequest
from ..manipulator import create_embedding_jsonl, create_completion_jsonl, get_running_names, get_duplicated_names, add_embeddings, fetch_and_construct, get_candidates_construct
from ..manipulator.df_constructor import build_similarity, select_top_n, get_candidates
from ..utils.utilities import MoveOriginalFolder


class Extraction:

    def __init__(self, from_: str, save_filepath: str, query_and_prompts: dict[str,str], embedding_limit: Tuple[float] = (3000, 1000000), completion_limit: Tuple[float] = (3000, 60000), max_attempts: int = 5, logging_level: int = 10):
        """Main api for llm Extraction, please ensure your environ has your openai api key. The processes include embedding, classification and summarise.

        :param from_: the file path to the articles
        :type from_: str
        :param save_filepath: the file path where you save results
        :type save_filepath: str
        :query_and_prompts: a dict contain query, system_message, prompt_cls, prompt_sum keys and correspond values
        :type query_and_prompts: dict[str, str]
        :param embedding_limit: a tuple (request_limit, token_limit)
        :type embedding_limit: Tuple[float]
        :param completion_limit: a tuple (request_limit, token_limit)
        :type completion_limit: Tuple[float]
        :param max_attempts: the maximum retries for single request
        :type max_attempts: int
        :param logging_level: logging level, defaults to 10
        :type logging_level: int, optional
        :param pydantic_model: pydantic model for parsing results
        :type pydantic_model: BaseModel, optional
        """

        self.from_ = from_
        self.save_filepath = save_filepath
        assert (len(embedding_limit), len(completion_limit)) == (2,2), "check your limit settings"
        self.max_requests_per_minute_eb = embedding_limit[0]
        self.max_tokens_per_minute_eb = embedding_limit[1]
        self.max_requests_per_minute_cp = completion_limit[0]
        self.max_tokens_per_minute_cp = completion_limit[1]
        self.max_attempts = max_attempts
        self.logging_level = logging_level
        self.query = query_and_prompts["query"]
        self.system_message = query_and_prompts["system_message"]
        self.prompt_cls = query_and_prompts["prompt_cls"]
        self.prompt_sum = query_and_prompts["prompt_sum"]

    @MoveOriginalFolder(folder_path="data")
    async def extract(self, save_folder="data", pydantic_model: BaseModel = None, sample_size: int = None):
        # httpx configs
        timeout = httpx.Timeout(10.0, connect=30.0, pool=30.0, read=30.0)
        limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)

        # connect to vector database
        chromadb_client = chromadb.PersistentClient()
        collection = chromadb_client.get_or_create_collection("chromadb")

        async with AsyncOpenAI(
            http_client=httpx.AsyncClient(verify=False, timeout=timeout, limits=limits),
            max_retries=0,
        ) as client:
            # instantiate messengers (fetch information back)
            embedding_messenger = EmbeddingRequest(
                client=client,
                max_requests_per_minute=self.max_requests_per_minute_eb,
                max_tokens_per_minute=self.max_tokens_per_minute_eb,
                max_attempts=self.max_attempts,
                logging_level=self.logging_level
            )
            completion_messenger = CompletionRequest(
                client=client,
                max_requests_per_minute=self.max_requests_per_minute_cp,
                max_tokens_per_minute=self.max_tokens_per_minute_cp,
                max_attempts=self.max_attempts,
                logging_level=self.logging_level
            )
            
            running_names = get_running_names(directory=self.from_, sample_size=sample_size)
            await self.update_chromadb(
                running_names=running_names,
                embedding_messenger=embedding_messenger,
                collection=collection,
                sample_size=sample_size
            )

            cls_jsonl = fetch_and_construct(
                chroma_collection=collection,
                search_query=self.query,
                running_names=running_names,
                system_message=self.system_message,
                prompt=self.prompt_cls
            )
            with open(cls_jsonl, encoding='utf-8') as file:
                g = iter(file)
                g, probe_size, stop_flag = self.choose_probe_size(g)
                next_start_capacity = await completion_messenger.completion_helper(
                    requests_generator=g,
                    save_filepath=os.path.join("data", "completion_cls_results.jsonl"),
                    probe_size=probe_size,
                    stop_flag=stop_flag
                )
            
            sum_jsonl = get_candidates_construct(os.path.join("data", "completion_cls_results.jsonl"), system_message=self.system_message, prompt=self.prompt_sum)
            with open(sum_jsonl, encoding='utf-8') as file:
                g = iter(file)
                g, probe_size, stop_flag = self.choose_probe_size(g)
                await completion_messenger.completion_helper(
                    probe_size=probe_size,
                    requests_generator=g,
                    save_filepath=self.save_filepath,
                    pydantic_model=pydantic_model,
                    start_capacity=next_start_capacity,
                    stop_flag=stop_flag
                )

    async def update_chromadb(self, running_names, embedding_messenger, collection, sample_size):
        """embedding unseen docs then update chroma database"""
        duplicated_names = get_duplicated_names(
            running_names=running_names,
            chroma_collection=collection
        )
        embedding_reqeust_file, has_content = create_embedding_jsonl(source=self.from_, duplicated_articles=duplicated_names, chunk_size=100, sample_size=sample_size)
        if has_content:
            with open(embedding_reqeust_file, encoding='utf-8') as file:
                g = iter(file)
                await embedding_messenger.embedding_helper(
                    requests_generator=g,
                    save_filepath=os.path.join("data", "embedding_results.jsonl")
                )

            # update vector db
            add_embeddings(chroma_collection=collection, result_file=os.path.join("data", "embedding_results.jsonl"))
    
    async def extract_fatal(self, request_json_file, save_filepath: str, pydantic_model: BaseModel = None):
        """Use this method when last request exit without finishing. Need to pass the file name contains requests"""
        # httpx configs
        timeout = httpx.Timeout(10.0, connect=30.0, pool=30.0, read=30.0)
        limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)

        async with AsyncOpenAI(
            http_client=httpx.AsyncClient(verify=False, timeout=timeout, limits=limits),
            max_retries=0,
        ) as client:
            completion_messenger = CompletionRequest(
                client=client,
                max_requests_per_minute=self.max_requests_per_minute_cp,
                max_tokens_per_minute=self.max_tokens_per_minute_cp,
                max_attempts=self.max_attempts,
                logging_level=self.logging_level
            )

            with open(request_json_file, encoding='utf-8') as file:
                g = iter(file)
                g, probe_size, stop_flag = self.choose_probe_size(g)
                
                await completion_messenger.completion_helper(
                    probe_size=probe_size,
                    requests_generator=g,
                    save_filepath=save_filepath,
                    pydantic_model=pydantic_model,
                    stop_flag=stop_flag
                )

    def choose_probe_size(self, g):
        """choose the probe size based on the length of an iterator and return a new one, stop_flag indicate small size input"""
        g_ls = list(g)
        length = len(g_ls)
        g = iter(g_ls)
        stop_flag: bool = False
        assert length != 0 ,"Please refactor your prompt. Check whether they are comply with your article content or consider to loosen the rules in prompt_cls"
        if length > 100: # indicate big data input
            probe_size = 30
        else:
            probe_size = 10 # default for completion_helper
        if length <= probe_size:
            stop_flag = True
        return g, probe_size, stop_flag
    
    @MoveOriginalFolder(folder_path="data")
    async def classify(self, save_folder="data", pydantic_model: BaseModel = None, sample_size: int = None):
        # httpx configs
        timeout = httpx.Timeout(10.0, connect=30.0, pool=30.0, read=30.0)
        limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)

        # connect to vector database
        chromadb_client = chromadb.PersistentClient()
        collection = chromadb_client.get_or_create_collection("chromadb")

        async with AsyncOpenAI(
            http_client=httpx.AsyncClient(verify=False, timeout=timeout, limits=limits),
            max_retries=0,
        ) as client:
            # instantiate messengers (fetch information back)
            embedding_messenger = EmbeddingRequest(
                client=client,
                max_requests_per_minute=self.max_requests_per_minute_eb,
                max_tokens_per_minute=self.max_tokens_per_minute_eb,
                max_attempts=self.max_attempts,
                logging_level=self.logging_level
            )
            completion_messenger = CompletionRequest(
                client=client,
                max_requests_per_minute=self.max_requests_per_minute_cp,
                max_tokens_per_minute=self.max_tokens_per_minute_cp,
                max_attempts=self.max_attempts,
                logging_level=self.logging_level
            )
            
            running_names = get_running_names(directory=self.from_, sample_size=sample_size)
            await self.update_chromadb(
                running_names=running_names,
                embedding_messenger=embedding_messenger,
                collection=collection,
                sample_size=sample_size
            )
            from sisyphus.manipulator.jsonl_constructor import create_completion_from_embedding
            cls_file = os.path.join("data", "classify.jsonl")
            cls_file_res = os.path.join("data", "classify_res.jsonl")
            create_completion_from_embedding(os.path.join("data", "embedding.jsonl"), cls_file, self.system_message, self.prompt_cls)
            with open(cls_file, encoding='utf-8') as file:
                g = iter(file)
                g, probe_size, stop_flag = self.choose_probe_size(g)
                next_start_capacity = await completion_messenger.completion_helper(
                    requests_generator=g,
                    save_filepath=cls_file_res,
                    probe_size=probe_size,
                    stop_flag=stop_flag
                )