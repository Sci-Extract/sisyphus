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
import json
import os
import time
import threading
import logging
import logging.config
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager


def load_config(path):
    config = json.loads(path.read_text(encoding='utf-8'))
    return config

sep = os.sep
CONFIG_PATH = Path(os.sep.join(["config", "throttle_config.json"]))
CONFIG = load_config(CONFIG_PATH)

logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')

# region chat
@dataclass
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

    cool_down_sentinel: bool = False
    cool_down_time: int = 10
    cool_down_start: float = None
    openai_api_429_hits: int = 0
    
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
                if self.cool_down_sentinel:
                    await self.cool_down()
                await asyncio.sleep(time_sleep)
                self.instill()
            self.consume(consumed_tokens)
            self.instill()

    def consume(self, tokens: int, completion_tokens=50): # set default completion for 50
        self.left_tokens -= (tokens + completion_tokens)

    def setter(self, left_tokens):
        """
        setter will delegate to thread
        """
        with self.t_lock:
            if left_tokens is not None:
                self.left_tokens = left_tokens
                self.last_check = time.time()
    
    def retry_callback(self, retry_state):
        """callback for tenacity retry when hit 429 rate error"""
        self.cool_down_sentinel = True
        self.cool_down_start = time.time()
        self.openai_api_429_hits += 1
        
        logger.debug('%s', retry_state.outcome.exception())

    async def cool_down(self):
        """cool down for `cool_down_time` since last hit error"""
        while True:
            since_cool_down = time.time() - self.cool_down_start
            if since_cool_down >= self.cool_down_time:
                self.cool_down_sentinel = False
                break
            await asyncio.sleep(0.1)


chat_throttler = ChatThrottler(CONFIG["chat"]["max_tokens"], CONFIG["chat"]["time_frame"])

@asynccontextmanager
async def chat_waiter(consumed_tokens: int):
    await chat_throttler.wait_capacity(consumed_tokens)
    yield
    # some clean steps...

# endregion

# region embed
class EmbedThrottler(ChatThrottler):
    """
    Implementation specific to embedding
    - no need to use setter due to the accurate token counting.
    """
    def consume(self, tokens: int, completion_tokens=0):
        return super().consume(tokens, completion_tokens)

embed_throttler = EmbedThrottler(CONFIG["embed"]["max_tokens"], CONFIG["embed"]["time_frame"])

@asynccontextmanager
async def embed_waiter(consumed_tokens: int):
    logger.debug(consumed_tokens)
    await embed_throttler.wait_capacity(consumed_tokens)
    yield
    # some clean steps...

# endregion