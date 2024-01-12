import os
from typing import Tuple

import httpx
from numpy import array
from openai import AsyncOpenAI
from pydantic import BaseModel

from .parallel_processor import EmbeddingRequest, CompletionRequest
from ..manipulator import create_embedding_jsonl, create_completion_jsonl
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

            embedding_reqeust_file = create_embedding_jsonl(source=self.from_, chunk_size=200, sample_size=sample_size)
            with open(embedding_reqeust_file, encoding='utf-8') as file:
                g = iter(file)
                await embedding_messenger.embedding_helper(
                    requests_generator=g,
                    save_filepath=os.path.join("data", "embedding_results.jsonl")
                )
                
            query_embedding = await client.embeddings.create(
            input=self.query,
            model="text-embedding-ada-002",
            )
            query_vector = array(query_embedding.data[0].embedding)
            df_selected = select_top_n(
                df=build_similarity(
                    os.path.join("data", "embedding_results.jsonl"),
                    standard_vector=query_vector,
                    save_file=None
                ),
                top_n=5,
                save_file=os.path.join("data", "text_selected.csv")
            )
            create_completion_jsonl(
                df=df_selected,
                file_path=os.path.join("data", "completion_cls.jsonl"),
                system_message=self.system_message,
                prompt=self.prompt_cls,
                required_format='json'
            )
            with open(os.path.join("data", "completion_cls.jsonl"), encoding='utf-8') as file:
                g = iter(file)
                g, probe_size, stop_flag = self.choose_probe_size(g)
                next_start_capacity = await completion_messenger.completion_helper(
                    requests_generator=g,
                    save_filepath=os.path.join("data", "completion_cls_results.jsonl"),
                    probe_size=probe_size,
                    stop_flag=stop_flag
                )
                
            df_to_extract = get_candidates(os.path.join("data", "completion_cls_results.jsonl"), os.path.join("data", "embedding.jsonl"))
            create_completion_jsonl(
                df=df_to_extract,
                file_path=os.path.join("data", "completion_sum.jsonl"),
                system_message=self.system_message,
                prompt=self.prompt_sum,
                required_format='json'
            )
            with open(os.path.join("data", "completion_sum.jsonl"), encoding='utf-8') as file:
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
                g, probe_size = self.choose_probe_size(g)
                
                await completion_messenger.completion_helper(
                    probe_size=probe_size,
                    requests_generator=g,
                    save_filepath=save_filepath,
                    pydantic_model=pydantic_model
                )

    def choose_probe_size(self, g):
        """choose the probe size based on the length of an iterator and return a new one, stop_flag indicate small size input"""
        g_ls = list(g)
        length = len(g_ls)
        g = iter(g_ls)
        stop_flag: bool = False
        if length > 100: # indicate big data input
            probe_size = 30
        else:
            probe_size = 10 # default for completion_helper
        if length < probe_size:
            stop_flag = True
        return g, probe_size, stop_flag