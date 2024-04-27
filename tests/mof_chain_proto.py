# -*- coding:utf-8 -*-
'''
@File    :   mof_chain.py
@Time    :   2024/04/24 22:49:55
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   prototyping for mof extraction, without validation
'''

import asyncio
import re
import glob
import argparse
from typing import Optional
import uuid
from typing import Dict, List, TypedDict

import tqdm
import chromadb
import pandas as pd
from langchain_community.vectorstores import chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import (BaseModel, Field, ValidationError)
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
    aembed_httpx_client
    )


embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
# db = chroma.Chroma("test", embedding_function=embedding, client=client)
# model = ChatOpenAIThrottle(temperature=0, model='gpt-4-turbo-2024-04-09', http_async_client=achat_httpx_client)
model = ChatOpenAIThrottle(temperature=0)

class ExtractUptake(BaseModel):
    """Extract uptake/adsorption information from text"""
    mof_name: str = Field(..., description="the entity of gas uptaking")
    gas_type: str = Field(..., description="the gas used in adsorption/uptake")
    uptake: str = Field(..., description="the quantity and correspond unit of the adsorption/uptake, e.g., 3 mmol/g")
    temperature: Optional[str] = Field(..., description="at which temperature, e.g., 298 K")
    pressure: Optional[str] = Field(..., description='at which pressure, e.g., 1 bar')

def get_relevancy(database, file_name, query):
    docs = database.similarity_search(
    query=query,
    filter={"source": file_name}
    )
    return docs

def regex_filter_chain(doc: Document) -> Document | None:
    """filter doc"""
    uptake_unit = re.compile(r'(cm3[/\s]g(-1)?)|(mmol[/\s]g(-1)?)', re.I)
    return doc if uptake_unit.search(doc.page_content) else None

def extract_chain_v2(doc, examples) -> list[BaseModel] | None:
    parser = PydanticToolsParser(tools=[ExtractUptake])
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a helpful assistant to extract information from text, if you are do not know the vlaue of an asked field, return null for the value. Call the function multiple times if needed'),
            MessagesPlaceholder(variable_name='examples'),
            ('human', '{text}')
        ]
    )
    chain = prompt | model.bind_tools(tools=[ExtractUptake], tool_choice='auto')
    messages = chain.invoke({'text': doc.page_content, 'examples': examples})
    # check whether has made function call
    if not messages.additional_kwargs:
        return
    return parser.invoke(messages)

async def aextract_chain(doc, examples, data_models: list[BaseModel]):
    parser = PydanticToolsParser(tools=data_models)
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a helpful assistant to extract information from text, if you are do not know the vlaue of an asked field, return null for the value. Call the function multiple times if needed'),
            MessagesPlaceholder(variable_name='examples'),
            ('human', '{text}')
        ]
    )
    chain = prompt | model.bind_tools(tools=data_models, tool_choice='auto')
    messages = await chain.ainvoke({'text': doc.page_content, 'examples': examples})
    # check whether has made function call
    if not messages.additional_kwargs:
        return
    ret = None
    # For simplicity, skip message which failed passing validation
    try:
        ret = await parser.ainvoke(messages)
    except ValidationError as e:
        pass
    return ret 


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

def full_chain(doc: Document, examples: list[BaseMessage]):
    filter_ = regex_filter_chain(doc)
    res = None
    if filter_:
        res = extract_chain_v2(doc, examples)
    return res

async def achain(doc, examples, data_models):
    filter_ = regex_filter_chain(doc)
    res = None
    if filter_:
        res = await aextract_chain(doc, examples, data_models)
    return res

async def supervisor(database, query, examples, data_models, batch_size):
    names_list = glob.glob("articles_processed\\*.html")
    names_list = [name.split('\\')[-1] for name in names_list]
    results = []
    for index in range(0, len(names_list), batch_size):
        coros = []
        for file_name in names_list[index: index+batch_size]:
            docs = get_relevancy(database, file_name, query)
            coros.extend([achain(doc, examples, data_models) for doc in docs])
        # results.extend(await asyncio.gather(*coros))
        for coro in tqdm.tqdm(asyncio.as_completed(coros), total=len(coros)):
            res = await coro
            results.append(res)
    valid_results = [result for result in results if result]
    return valid_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collection_name",
        help="The collection name of the vector database, be sure that you have indexed it already"
    )
    args = parser.parse_args()

    client = chromadb.HttpClient()
    db = chroma.Chroma(args.collection_name, embedding_function=embedding, client=client)
    tool_examples = [
    ('Example: The single-component isotherms revealed that [Cd2(dpip)2(DMF)(H2O)]·DMF·H2O adsorbs '
     '124.4/182.8 cm3 g−1 of C2H2, 76.8/120.0 cm3 g−1 of C2H4 '
     'at 298 and 273 K under 100 kPa, respectively.',
    [ExtractUptake(mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O', gas_type='C2H2', uptake='124.4 cm3/g', temperature='298 K', pressure='100 kPa'),
    ExtractUptake(mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O', gas_type='C2H2', uptake='182.8 cm3/g', temperature='273 K', pressure='100 kPa'),
    ExtractUptake(mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O', gas_type='C2H4', uptake='76.8 cm3/g', temperature='298 K', pressure='100 kPa'),
    ExtractUptake(mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O', gas_type='C2H4', uptake='120.0 cm3/g', temperature='273 K', pressure='100 kPa')])]
    input, tool_calls = tool_examples[0]
    query = "single-component adsorption isotherms, uptake value are 3 mmol g-1 at 298 K 1 bar"
    examples = tool_example_to_messages({"input": input, "tool_calls": tool_calls})
    res = asyncio.run(supervisor(db, query, examples, [ExtractUptake], 10))
    data_dic = []
    for models in res:
        for model in models:
            model: BaseModel
            data_dic.append(dict(model))
    df = pd.DataFrame(data_dic)
    df.to_csv('mof_50.csv', index=False)

if __name__ == '__main__':
    main()
