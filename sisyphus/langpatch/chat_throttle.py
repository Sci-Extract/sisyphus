# -*- coding:utf-8 -*-
'''
@File    :   throttler.py
@Time    :   2024/04/15 20:57:44
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Trottlling openai chat model
'''

import asyncio
import dataclasses
import functools
import json
import time
import threading
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager

import httpx
from langchain_community.callbacks import get_openai_callback


CONFIG_PATH = Path("throttle_config.json") 
def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    return config

CONFIG = load_config()

@dataclasses
class ChatThrottler:
    """
    Implementation of throttler
    - setter() is called after a httpx response
    - both instill() and setter() can change self.left_tokens
    """
    max_tokens: int
    time_frame: Literal["s", "m"]
    last_check: float = None
    a_lock: asyncio.Lock = asyncio.Lock()
    t_lock: threading.Lock = threading.Lock()  
    
    def __post_init__(self):
        self.left_tokens: float
        self.instill_rate: float
        self.left_tokens = self.max_tokens
        self.reqeust_instill_rate: float
        if self.time_frame == "m":
            self.token_instill_rate = self.max_tokens / 60
        else:
            self.token_instill_rate = self.max_tokens
         
    def instill(self):
        current = time.time()
        elapsed = current - self.last_check
        self.left_tokens = min(self.max_tokens, self.left_tokens + elapsed * self.token_instill_rate)
        self.last_check = current

    async def wait_capacity(self, consumed_tokens: int, time_sleep=0.1):
        """
        Used for throttling.
        - Blocking until there is enough capacity, nonbloking when sufficiency.
        """
        if not self.last_check:
            self.last_check = time.time()
        async with self.a_lock:
            while (consumed_tokens - self.left_tokens) > 0:
                await asyncio.sleep(time_sleep)
                self.instill()
            self.consume(consumed_tokens)
            self.instill()

    def consume(self, tokens: int):
        self.left_tokens -= tokens

    def setter(self, left_tokens):
        """
        setter will delegate to thread
        """
        with self.t_lock:
            if left_tokens is not None:
                self.left_tokens = left_tokens
                self.last_check = time.time()

throttler = ChatThrottler(CONFIG["max_tokens"], CONFIG["time_frame"])

@asynccontextmanager
async def waiter(throttler: ChatThrottler, consumed_tokens: int):
    yield await throttler.wait_capacity(consumed_tokens)
    # some clean steps...

async def httpx_response_hooker(throtter: ChatThrottler, response: httpx.Response):
    assert "x-ratelimit-limit-requests" in response.headers # for test
    left_tokens = response.headers.get("x-ratelimit-remaining-tokens", None)
    await asyncio.to_thread(throtter.setter, left_tokens)
    
timeout = httpx.Timeout(10.0, connect=30.0, pool=30.0, read=30.0)
limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
httpx_client = httpx.AsyncClient(
    timeout=timeout,
    limits=limits,
    verify=False,
    event_hooks={"response": [functools.partial(httpx_response_hooker, throttler)]}
)
