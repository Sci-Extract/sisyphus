"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python examples/api_request_parallel_processor.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-ada-002", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-ada-002", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-ada-002`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 1, 2, 3, ...)
    - Run main()
"""

# imports

import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import os  # for reading API key
import re  # for matching endpoint from request URL
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
from logging import Logger
from typing import Generator, IO, Literal, Optional

import httpx  # for making API calls concurrently
import openai
import pydantic
import tiktoken  # for counting tokens
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..utils.utilities import log # for logging rate limit warnings and other messages

# pydantic models
#from mof_absorb_pydantic import Compounds

@dataclass
class ProcessRequest:
    """Base class for request process, implementation by inheritance"""

    client: AsyncOpenAI
    max_requests_per_minute: float
    max_tokens_per_minute: float
    max_attempts: int
    token_encoding_name: str = "cl100k_base"
    logging_level: int = 10 # for debug
    
    def __post_init__(self):
        self.logger = log(logging_level=self.logging_level)

    async def _process_request(self, requests_generator: Generator, save_filepath: str, mode: Literal["embeddings", "completions"], bucket: "Bucket", requests_rate: float = 0.001, probe_size: int = 0, completion_tokens: int = 15, record_usage: bool = False, pydantic_model: BaseModel = None):
        """Process requests parallelly. For completion task, set probe_size to get average token consumption. for simple request, set record_usage to False (note that it only returns the remain token, not sleep to pause execution)"""

        if mode not in ['embeddings', 'completions']:
            raise ValueError("mode only have two choices: embeddings/completions")

        seconds_to_pause_after_rate_limit_error = 15

        self.logger.debug(f"Logging initialized at level {self.logging_level}")

        queue_of_requests_to_retry = asyncio.Queue()
        task_id_generator = self.task_id_generator()
        status_tracker = StatusTracker()
        completion_token_usage = TokenUsage() if record_usage or probe_size else None
        next_request = None
        file_not_finished = True

        self.logger.debug(f"Initialization complete. Present at {mode} mode")
        
        while True:
            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if (
                seconds_since_rate_limit_error
                < seconds_to_pause_after_rate_limit_error
            ):
                remaining_seconds_to_pause = (
                    seconds_to_pause_after_rate_limit_error
                    - seconds_since_rate_limit_error
                )
                self.logger.warn(
                    f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                )
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago

            # check if there are remain requests
            if next_request is None:
                if file_not_finished:
                    try:
                        next_g = next(requests_generator)
                        if isinstance(next_g, dict):
                            request_info = next_g
                        else:
                            request_info = json.loads(next_g) # the raw json according with openai request format, can include metadata field.
                        token_consumption = num_tokens_consumed_from_request(request_info, mode, self.token_encoding_name, max_tokens=completion_tokens)
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_info,
                            token_consumption=token_consumption,
                            attempts_left=self.max_attempts,
                            metadata=request_info.pop("metadata", None),
                            logger=self.logger,
                            pydantic_model=pydantic_model,
                            completion_token_usage=completion_token_usage
                        )
                        status_tracker.num_tasks_in_progress += 1
                        status_tracker.num_tasks_started += 1

                        # when setting probe
                        if probe_size == status_tracker.num_tasks_started:
                            raise StopIteration
                        
                    except StopIteration:
                        file_not_finished = False
                        self.logger.debug("Reads all data")

                elif not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
            
            # update available capacity
            bucket.update_capacity()

            if next_request:
                if bucket.has_capacity(next_request.token_consumption):
                    asyncio.create_task(
                        next_request.call_api(
                            client=self.client,
                            mode=mode,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker
                        )
                    )
                    bucket.set_capacity(next_request.token_consumption)
                    next_request.attempts_left -= 1
                    next_request = None
            
            if status_tracker.num_tasks_in_progress == 0:
                
                break

            # sleep in main loop so that task can run
            await asyncio.sleep(requests_rate)

        # after finishing, log final status
        self.logger.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            self.logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
            )
        if status_tracker.num_rate_limit_errors > 0:
            self.logger.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )
        if status_tracker.num_task_validate_errors > 0:
            self.logger.warning(
                f"{status_tracker.num_task_validate_errors} / {status_tracker.num_tasks_started} requests failed with validation."
            )

        if completion_token_usage:
            # assert len(completion_token_usage.completion_tokens) == probe_size, f'something went wrong, {completion_token_usage.completion_tokens}'
            completion_tokens_average = sum(completion_token_usage.completion_tokens) / len(completion_token_usage.completion_tokens) # the average token in completion
            remain_tokens = completion_token_usage.x_ratelimit_remaining_tokens[-1] # the remain tokens of last time stamp
            remain_requests = completion_token_usage.x_ratelimit_remaining_requests[-1]
            last_time_stamp = completion_token_usage.last_time_stamp[-1]
            self.logger.warning(f"OPENAI quota: remaining requests: {remain_requests}, remaining tokens: {remain_tokens}")
            return (last_time_stamp, remain_requests, remain_tokens, completion_tokens_average) if probe_size else (last_time_stamp, remain_requests, remain_tokens)

    def task_id_generator(self):
        """Generate integers 0, 1, 2, and so on."""
        task_id = 0
        while True:
            yield task_id
            task_id += 1
        

class EmbeddingRequest(ProcessRequest):
    """Making requests for embedding process"""
    mode = 'embeddings'
    async def embedding_helper(self, requests_generator: Generator, save_filepath: str):
        """Implementation method"""
        bucket = Bucket(last_update_time=time.time(), max_capacity_requests=self.max_requests_per_minute, max_capacity_tokens=self.max_tokens_per_minute)
        await self._process_request(requests_generator=requests_generator, save_filepath=save_filepath, mode=self.mode, bucket=bucket)


class CompletionRequest(ProcessRequest):
    """Making requests for completions"""
    mode = 'completions'
    async def completion_helper(self, requests_generator: Generator, save_filepath: str, probe_size: int = 10, pydantic_model: BaseModel = None, start_capacity: tuple = None, stop_flag: bool = False):
        """Implementation method, return last timestamp of api call and  api left request, token quota"""
        if start_capacity:
            last_time_stamp, remain_requests, remain_tokens = start_capacity
            probe_bucket = Bucket(last_update_time=last_time_stamp, max_capacity_requests=self.max_requests_per_minute, max_capacity_tokens=self.max_tokens_per_minute)
            probe_bucket.update_primal_capacity(remain_requests, remain_tokens)
        else:
            probe_bucket = Bucket(last_update_time=time.time(), max_capacity_requests=self.max_requests_per_minute, max_capacity_tokens=self.max_tokens_per_minute)
        
        # Probe process
        probe_last_time_stamp, probe_remain_requests, probe_remain_tokens, completion_tokens_ave = await self._process_request(
            requests_generator=requests_generator,
            save_filepath=save_filepath,
            mode=self.mode, bucket=probe_bucket,
            probe_size=probe_size,
            pydantic_model=pydantic_model,
            record_usage=True,
        )
        self.logger.warning(f"The estimated token completion consumption is set to {completion_tokens_ave} tokens")
        
        if stop_flag:
            self.logger.info("Maybe you are running test since the input data is small")
            return probe_last_time_stamp, probe_remain_requests, probe_remain_tokens

        # process rest requests
        bucket = Bucket(last_update_time=probe_last_time_stamp, max_capacity_requests=self.max_requests_per_minute, max_capacity_tokens=self.max_tokens_per_minute)
        # update start capacity
        bucket.update_primal_capacity(remain_requests=probe_remain_requests, remain_tokens=probe_remain_tokens)

        f_last_time_stamp, f_remain_requests, f_remain_tokens = await self._process_request(
            requests_generator=requests_generator,
            save_filepath=save_filepath, 
            mode=self.mode, 
            bucket=bucket, 
            completion_tokens=completion_tokens_ave,
            pydantic_model=pydantic_model,
            record_usage=True,
        )

        # final time, requests and tokens quota.
        return f_last_time_stamp, f_remain_requests, f_remain_tokens

    async def completion_helper_with_no_probe(self, requests_generator: Generator, save_filepath: str, pydantic_model: BaseModel = None):
        """For small tasks use this method"""

        bucket = Bucket(last_update_time=time.time(), max_capacity_requests=self.max_requests_per_minute, max_capacity_tokens=self.max_tokens_per_minute)
        await self._process_request(
            requests_generator=requests_generator,
            save_filepath=save_filepath, 
            mode=self.mode, 
            bucket=bucket, 
            pydantic_model=pydantic_model,
        )
            
    async def token_rehabilitation(self, remain_tokens):
        if (token_to_restore := self.max_tokens_per_minute - remain_tokens) > 0:
            time_to_sleep = token_to_restore / self.max_tokens_per_minute * 60
            self.logger.warning(f"token rehabilitation for {time_to_sleep} s")
            await asyncio.sleep(time_to_sleep)
            


@dataclass
class Bucket:
    """GCRA implementation, update capacity"""
    last_update_time: float
    max_capacity_requests: float
    max_capacity_tokens: float

    def __post_init__(self):
        self.present_capacity_requests = self.max_capacity_requests
        self.present_capacity_tokens = self.max_capacity_tokens
    
    def update_primal_capacity(self, remain_requests: float, remain_tokens: float):
        """Successive requests start with a smaller capacity"""
        self.present_capacity_requests = remain_requests
        self.present_capacity_tokens = remain_tokens
    
    def update_capacity(self):
        self.current_time = time.time()
        elapse = self.current_time - self.last_update_time
        self.present_capacity_requests = min(self.present_capacity_requests + elapse * self.max_capacity_requests / 60,
                                    self.max_capacity_requests)
        self.present_capacity_tokens = min(self.present_capacity_tokens + elapse * self.max_capacity_tokens / 60,
                                    self.max_capacity_tokens)
        self.last_update_time = self.current_time
    
    def set_capacity(self, value):
        self.present_capacity_requests -= 1
        self.present_capacity_tokens -= value

    def has_capacity(self, value):
        return True if (self.present_capacity_requests > 1 and self.present_capacity_tokens > value) else False

    def get_last_update_time(self):
        return self.last_update_time
    
    def get_capacities(self):
        return self.present_capacity_requests, self.present_capacity_tokens

# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_task_validate_errors: int = 0 # pydantic validation errors
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

@dataclass
class TokenUsage():
    """Track completion token usage"""
    # returned by openai: "usage": {"prompt_tokens": 437, "completion_tokens": 20, "total_tokens": 457}
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    total_tokens: list[int] = field(default_factory=list)
    
    x_ratelimit_remaining_tokens: list[int] = field(default_factory=list)
    x_ratelimit_remaining_requests: list[int] = field(default_factory=list)
    last_time_stamp: list[float] = field(default_factory=list)


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    logger: Logger
    pydantic_model: Optional[BaseModel]
    result: list = field(default_factory=list)
    completion_token_usage: Optional[TokenUsage] = None

    async def call_api(
        self,
        client: AsyncOpenAI,
        mode: Literal['embeddings', 'completions'],
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        self.logger.info(f"Starting request #{self.task_id}")
        error = None
        try:                                                                                                                                                                                                         
            if mode == 'embeddings':
                response = await client.embeddings.create(**self.request_json) # note that json metadata filed has been poped out
                llm_result = dict(embedding=response.data[0].embedding)
                
            elif mode == 'completions':
                raw_response = await client.chat.completions.with_raw_response.create(**self.request_json)
                response = raw_response.parse()
                llm_result = dict(content=json.loads(response.choices[0].message.content))
                # if pass in pydantic model then validate the llm result
                if self.pydantic_model:
                    content = llm_result["content"]
                    model = self.pydantic_model.model_validate(content)
                    llm_result = dict(content=model.model_dump())
            else:
                raise ValueError('choose between embeddings/completions')
            
        except openai.APIConnectionError as e:
            status_tracker.num_api_errors += 1
            self.logger.warning(f"/APICONNECTION/ Request {self.task_id}: The server could not be reached")
            self.logger.warning(e)
            self.logger.warning(e.__cause__)  # an underlying Exception, likely raised within httpx.
            error = e

        except openai.RateLimitError as e:
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
            self.logger.warning(f"/RATELIMIT/ Request {self.task_id}: a 429 status code was received; we should back off a bit.")
            error = e
            
        except openai.APIStatusError as e:
            status_tracker.num_api_errors += 1
            self.logger.warning(f"/APISTATUS/ Request {self.task_id}: another non-200-range status code was received")
            self.logger.warning(e.status_code)
            self.logger.warning(e.response)
            error = e
        
        except pydantic.ValidationError as e:
            status_tracker.num_task_validate_errors += 1
            self.logger.warning(f"/VALIDATION/ Request {self.task_id}: {e.json()}")
            error = e
            
        except (Exception) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            self.logger.warning(f"Request {self.task_id}: failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                self.logger.error(
                    f"Request [{self.task_id}]: failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, "Failed", self.metadata]
                    if self.metadata
                    else [self.request_json, "Failed"]
                )
                self.append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
                
        # when success
        else:
            # real token usage for probe
            if self.completion_token_usage:
                self.completion_token_usage.x_ratelimit_remaining_tokens.append(int(raw_response.headers.get('x-ratelimit-remaining-tokens')))
                self.completion_token_usage.x_ratelimit_remaining_requests.append(int(raw_response.headers.get('x-ratelimit-remaining-requests')))
                self.completion_token_usage.last_time_stamp.append(time.time())
                self.completion_token_usage.prompt_tokens.append(response.usage.prompt_tokens)
                self.completion_token_usage.completion_tokens.append(response.usage.total_tokens - response.usage.prompt_tokens) # usage has no completion tokens attribute, calculate the difference.
            data = (
                [self.request_json, llm_result, self.metadata]
                if self.metadata
                else [self.request_json, llm_result]
            )
            self.append_to_jsonl(data, save_filepath)

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            self.logger.debug(f"Request {self.task_id} saved to {save_filepath}")

    def append_to_jsonl(self, data, filename: str) -> None:
        """Append a json payload to the end of a jsonl file."""
        json_string = json.dumps(data)
        with open(filename, "a") as f:
            f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    mode: str,
    token_encoding_name: str,
    max_tokens: int = 15,
    n: int = 1
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if mode == "completions":
        completion_tokens = n * max_tokens
        num_tokens = 0
        for message in request_json["messages"]:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens + completion_tokens
        
    # if embeddings request, tokens = input tokens
    elif mode == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'Please provide the right mode, either completions or embeddings'
        )
