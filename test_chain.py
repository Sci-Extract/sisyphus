import argparse
import warnings
import time
import asyncio
import os
import logging
import re
from typing import Optional

import chromadb
from langchain_core.documents import Document
from langchain.pydantic_v1 import BaseModel, Field
from sqlmodel import SQLModel, Relationship, create_engine

from sisyphus.chain.database import  extend_fields, ResultDB
from sisyphus.chain import (
    Chain,
    Filter,
    Extractor,
    Validator,
    Writer,
    asupervisor,
)
from sisyphus.chain.chain_elements import DocDB
from sisyphus.patch import (
    ChatOpenAIThrottle,
    OpenAIEmbeddingThrottle,
    achat_httpx_client,
    aembed_httpx_client,
    AsyncChroma,
)
from sisyphus.chain.chain_elements import run_chains
from example_helper import tool_example_to_messages


logging.basicConfig(level=logging.WARNING)

# constants
DEFAULT_DIR = 'db'

# helper functions
def create_chain(
    engine,
    model,
    examples,
    doc_db,
    query,
    filter_func,
    pydantic_model,
):
    """create extract chain"""
    filter = Filter(db=doc_db, query=query, filter_func=filter_func)
    extractor = Extractor(
        model, pydantic_models=[pydantic_model], examples=examples
    )
    validator = Validator()
    result_db = ResultDB(engine, pydantic_model)
    result_db.create_db()
    writer = Writer(result_db)
    chain = Chain(filter, extractor, validator, writer)
    return chain


def create_db_schema(save_as, default_dir='db'):
    sql_path = os.path.join(default_dir, save_as)
    engine = create_engine('sqlite:///' + sql_path)
    SQLModel.metadata.create_all(engine)
    return engine

#### Configurable section #####
# example of filter, remember that customizable functions must conform with this schema
def regex_filter(doc: Document) -> bool:
    uptake_unit = re.compile(r'(cm3[/\s]g(-1)?)|(mmol[/\s]g(-1)?)', re.I)
    return bool(uptake_unit.search(doc.page_content))


# example of pydantic model
class ExtractUptake(BaseModel):
    """Extract uptake/adsorption properties from text"""

    mof_name: str = Field(..., description='the entity of gas uptaking, ')
    gas_type: str = Field(..., description='the gas used in adsorption/uptake')
    uptake: float = Field(
        ...,
        description='the quantity of the adsorption/uptake, e.g., 3',
    )
    uptake_unit: str = Field(
        ..., description='the unit of uptake, e.g., mmol/g'
    )
    temperature: Optional[float] = Field(
        None, description='process at which temperature'
    )
    temperature_unit: Optional[str] = Field(
        None, description='unit of temperature, e.g. "K"'
    )
    pressure: Optional[float] = Field(
        None, description='process at which pressure'
    )
    pressure_unit: Optional[str] = Field(
        None, description='unit of pressure, e.g., "KPa"'
    )


# NOTE: I recommand you to provide examples to get result match to provided.
tool_examples = [
    (
        'Example: The single-component isotherms revealed that [Cd2(dpip)2(DMF)(H2O)]·DMF·H2O adsorbs '
        '124.4/182.8 cm3 g−1 of C2H2, 76.8/120.0 cm3 g−1 of C2H4 '
        'at 298 and 273 K under 100 kPa, respectively.',
        [
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H2',
                uptake=124.4,
                uptake_unit='cm3/g',
                temperature=298,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H2',
                uptake=182.8,
                uptake_unit='cm3/g',
                temperature=273,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H4',
                uptake=76.8,
                uptake_unit='cm3/g',
                temperature=298,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
            ExtractUptake(
                mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                gas_type='C2H4',
                uptake=120.0,
                uptake_unit='cm3/g',
                temperature=273,
                temperature_unit='K',
                pressure=100,
                pressure_unit='Kpa',
            ),
        ],
    )
]
#### End #####


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Main interface of sisyphus v2 /beta'
    )
    arg_parser.add_argument('-d', '--directory', required=True)
    arg_parser.add_argument('--save_as', default='result.db', required=False)
    arg_parser.add_argument('-c', '--collection_name', required=True)
    arg_parser.add_argument('-q', '--query', required=False)
    arg_parser.add_argument('-b', '--batch_size', default=10)
    args = arg_parser.parse_args()

    model = ChatOpenAIThrottle(http_async_client=achat_httpx_client)
    embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
    client = chromadb.HttpClient()
    db = AsyncChroma(
        collection_name=args.collection_name,
        embedding_function=embedding,
        client=client,
    )
    # or db without vectors
    # change the name to your own database
    # db = DocDB(engine='sqlite:///db\\plain.db')
    if isinstance(db, DocDB):
        warnings.warn(
            '>>> finding that you are running program using plain sql base, this may result into too many calls to openai. <<<'
        )
        user_input = input('continue, y/n: \n')
        if user_input == 'y':
            pass
        else:
            raise SystemExit
    
    examples = []
    for input_, tool_calls in tool_examples:
        examples.extend(tool_example_to_messages(
            {'input': input_, 'tool_calls': tool_calls}
        ))

    sql_path = os.path.join(DEFAULT_DIR, args.save_as)
    engine = create_engine('sqlite:///' + sql_path)

    chain = create_chain(
        engine=engine,
        model=model,
        examples=examples,
        doc_db=db,
        query=args.query,
        filter_func=regex_filter,
        pydantic_model=ExtractUptake,
    )

    batch_size = int(args.batch_size)
    # asyncio.run(asupervisor(chain, args.directory, batch_size))
    asyncio.run(run_chains(chain, args.directory, batch_size, namespace='mof/uptake'))
    print('hits 429 times:', model._chat_throttler.openai_api_429_hits) # To test retry logic
