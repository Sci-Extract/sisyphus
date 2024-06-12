import asyncio
import logging

from langchain.pydantic_v1 import BaseModel, Field

from sisyphus.index import create_vectordb_in_memory
from sisyphus.chain import Filter, Extractor, Validator, Writer
from sisyphus.utils.helper_functions import get_chat_model, get_create_resultdb


logging.basicConfig(level=20)

class ExtractSHG(BaseModel):
    """extract nlo material's SHG(second harmonic generation) coeffecients from text"""
    nlo_name: str = Field(description='the chemical name of the nlo material')
    shg: float = Field(description='the value of the shg coeffiencient, usually represented as 0.4 pm/V or times of KDP/AgGaS2')
    shg_unit: str = Field(description='the unit of the shg, pm/V or standard material like KDP/AgGaS2')

test_db = create_vectordb_in_memory(target_file='test.html', collection_name='test')
chat_model = get_chat_model()
result_db = get_create_resultdb('test', ExtractSHG)

chain = (Filter(test_db, query='second harmonic generation pm/V') +
         Extractor(chat_model, ExtractSHG) +
         Validator() +
         Writer(result_db)
)
asyncio.run(chain.acompose('test.html'))
