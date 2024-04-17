# -*- coding:utf-8 -*-
'''
@File    :   chat_patch.py
@Time    :   2024/04/17 16:41:53
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   patch chat model
'''

from typing import Any, Coroutine, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.language_models.chat_models import agenerate_from_stream

from sisyphus.langpatch.chat_throttle import throttler, waiter


class ChatOpenAIThrottle(ChatOpenAI):
    """
    Patch langchain chatopenai, use anywhere else inside this project as substitution of `ChatOpenAI`.
    """
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        async with waiter(throttler=throttler, consumed_tokens=self.get_num_tokens_from_messages(messages)):
            await super()._agenerate(messages, stop, run_manager, **kwargs)

