import argparse
import asyncio
import os
import re
from typing import Optional

import chromadb
from langchain_core.documents import Document
from langchain.pydantic_v1 import BaseModel, Field
from sqlmodel import SQLModel, Relationship, create_engine

from sisyphus.chain.database import DocBase, update_resultbase
from sisyphus.chain import (
    Chain,
    Filter,
    Extractor,
    Validator,
    SqlWriter,
    asupervisor,
)
from sisyphus.patch import (
    ChatOpenAIThrottle,
    OpenAIEmbeddingThrottle,
    achat_httpx_client,
    aembed_httpx_client,
    AsyncChroma,
)
from example_helper import tool_example_to_messages


def create_chain(
    engine, doc_orm, result_orm, model, examples, vector_db, query, filter_func, pydantic_model
):
    """create extract chain"""
    filter = Filter(db=vector_db, query=query, filter_func=filter_func)
    extractor = Extractor(
        model, pydantic_models=[pydantic_model], examples=examples
    )
    validator = Validator()
    writer = SqlWriter(engine, doc_orm, result_orm)
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
    """Extract uptake/adsorption information from text"""

    mof_name: str = Field(..., description='the entity of gas uptaking')
    gas_type: str = Field(..., description='the gas used in adsorption/uptake')
    uptake: str = Field(
        ...,
        description='the quantity and correspond unit of the adsorption/uptake, e.g., 3 mmol/g',
    )
    temperature: Optional[str] = Field(
        ..., description='at which temperature, e.g., 298 K'
    )
    pressure: Optional[str] = Field(
        ..., description='at which pressure, e.g., 1 bar'
    )


#### End #####


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Main interface of sisyphus v2'
    )
    arg_parser.add_argument('-c', '--collection_name', required=True)
    arg_parser.add_argument('-q', '--query', required=False)
    arg_parser.add_argument('--save_as', default='result.db', required=False)
    arg_parser.add_argument('-d', '--directory', required=True)
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

    class Doc(DocBase, table=True):
        __tablename__ = 'document'
        section: str
        results: Optional[list['Result']] =  Relationship(back_populates='doc')

    ResultBase = update_resultbase(ExtractUptake)

    class Result(ResultBase, table=True):
        doc: Doc = Relationship(back_populates='results')
    
    engine = create_db_schema(args.save_as)

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
                    uptake='124.4 cm3/g',
                    temperature='298 K',
                    pressure='100 kPa',
                ),
                ExtractUptake(
                    mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                    gas_type='C2H2',
                    uptake='182.8 cm3/g',
                    temperature='273 K',
                    pressure='100 kPa',
                ),
                ExtractUptake(
                    mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                    gas_type='C2H4',
                    uptake='76.8 cm3/g',
                    temperature='298 K',
                    pressure='100 kPa',
                ),
                ExtractUptake(
                    mof_name='[Cd2(dpip)2(DMF)(H2O)]·DMF·H2O',
                    gas_type='C2H4',
                    uptake='120.0 cm3/g',
                    temperature='273 K',
                    pressure='100 kPa',
                ),
            ],
        )
    ]
    input, tool_calls = tool_examples[0]
    examples = tool_example_to_messages(
        {'input': input, 'tool_calls': tool_calls}
    )
    chain = create_chain(
        engine,
        Doc,
        Result,
        model,
        examples,
        db,
        args.query,
        regex_filter,
        ExtractUptake,
    )
    asyncio.run(asupervisor(chain, args.directory, args.batch_size))
