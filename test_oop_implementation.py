import asyncio
import argparse

from sisyphus.chain.chain_elements import (
    Filter,
    Extractor,
    Validator,
    Writer,
    run_chains_with_extraction_history,
)
from sisyphus.chain.validators import coercion_validator

from sisyphus.utils.helper_functions import (
    get_chat_model,
    get_create_resultdb,
    get_remote_chromadb,
    get_plain_articledb,
    tool_example_to_messages
)

from model_with_examples import (
    regex_filter,
    ExtractUptake,
    tool_examples,
)

collection_name = 'mof' # the name of the chroma db
query = 'single-component adsorption isotherms, uptake value are 3 mmol g-1 at 298 K 1 bar'
db_name = 'oop_impl.db' # the name of the saved db
article_dir = 'articles_processed' # where your aritlces stores

chat_model = get_chat_model('gpt-3.5-turbo')
db = get_remote_chromadb(collection_name)
# db = get_plain_articledb('plain.db')
result_db = get_create_resultdb('oop_impl.db', ExtractUptake)
examples = []
for input_, tool_calls in tool_examples:
    examples.extend(
        tool_example_to_messages({'input': input_, 'tool_calls': tool_calls})
    )

filter_ = Filter(db, query, regex_filter)
extractor = Extractor(chat_model, ExtractUptake, examples)
validator = Validator()
validator.add_gadget(coercion_validator(['gas_type']))
writer = Writer(result_db)

chain = filter_ + extractor + validator + writer

asyncio.run(
    run_chains_with_extraction_history(chain, article_dir, 10, 'mof/uptake_4o')
)
