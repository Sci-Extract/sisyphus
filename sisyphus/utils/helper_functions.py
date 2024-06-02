"""
author: soike
date: 6/2/2024
description: provide easy use functions when creating `Chain` object
"""

import os
import uuid
from typing import Literal, List, TypedDict

import chromadb
from sqlmodel import create_engine
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    ChatOpenAIThrottle,
    achat_httpx_client,
    aembed_httpx_client,
    AsyncChroma,
)
from sisyphus.chain.database import ResultDB
from sisyphus.index.indexing import DEFAULT_DB_DIR, DocDB

def get_remote_chromadb(collection_name: str):
    """helper function to easily get chroma db using http protocal"""
    embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
    chromadb_client = chromadb.HttpClient()
    db = AsyncChroma(
        collection_name=collection_name,
        client=chromadb_client,
        embedding_function=embedding,
    )
    return db

def get_plain_articledb(db_name):
    """helper function to easily get plain db (db without embedding vectors)"""
    db_url = 'sqlite:///' + os.path.join(DEFAULT_DB_DIR, db_name)
    return  DocDB(create_engine(db_url))


def get_chat_model(
    model_name: Literal['gpt-3.5-turbo', 'gpt-4o'] = 'gpt-3.5-turbo'
):
    """helper function to easily get specified openai model, default to gpt-3.5-turbo"""
    return ChatOpenAIThrottle(
        http_async_client=achat_httpx_client, model_name=model_name
    )


def get_create_resultdb(db_name, pydantic_model: BaseModel, default_dir='db'):
    """helper function to easily get and create result database"""
    result_db = ResultDB(
        create_engine('sqlite:///' + os.path.join(default_dir, db_name)),
        pydantic_model,
    )
    result_db.create_db()
    return result_db


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted

def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages
    