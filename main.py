import asyncio
import time

from dotenv import find_dotenv, load_dotenv

from sisyphus.processor.llm_extraction import Extraction

from parse_result_absorb import Compounds

_ = load_dotenv(find_dotenv())

from_ = "data_articles"
system_message = "You are reading a piece of text from chemistry domain of MOF absorption, answer the question provided by user"
query = "MOF adsorption and desorption isotherms for Xe and Kr under a specific temperature and pressure (e.g., 298k, 1 bar), using uptake to measure the ability (e.g., 2.44 mmol g-1; 3.6 cm3/cm3)"

prompt_cls = \
"""Given a text quoted by triple backticks, judge whether the text contains the desired information. Return a JSON object with the following criteria:

a. Check if the text includes MOF compounds.
b. Have the temperature and pressure parameters describing the gas uptaking
c. Verify if the text includes mof uptake property for Xe or Kr in numerical form, e.g., 0.71 mmol g−1

Sequentially examine each criterion. If any criterion is not met, return False for that criterion. Output a JSON complying with the schema:

{
  "a": true/false,
  "b": true/false,
  "c": true/false
}

"""

prompt_sum = \
"""Given a text quoted by triple backticks containing information about MOF Xe/Kr absorption.
Extract only MOF and corresponding gas uptake with numerical values in JSON format.
The parameters of the uptake should be recorded, include the temperature, pressure and the gas name (usually Xe or Kr)
If multiple uptake values found, recording parameters accordingly .

The JSON format should adhere to the following structure:

{
  "compounds":[
    {
      "mof": <name>,
      "uptake": {
        "temperature": {"value": <value>, "unit": <unit>},
        "pressure": {"value": <value>, "unit": <unit>},
        "gas": <gas_name>,
        "result": {"value": <value>, "unit": <unit>},
      }
    },
    // Additional uptake values followed
  ]
}

the uptake value should be numerical value, descriptions as 'higher', 'lower', 'more than' are not allowed.
Filled with null if any key value is not found.

"""
d = dict(query=query, prompt_cls=prompt_cls, prompt_sum=prompt_sum, system_message=system_message)

start = time.perf_counter()
extraction = Extraction(from_=from_, save_filepath="results.jsonl", query_and_prompts=d, embedding_limit=(5000, 1000000), completion_limit=(5000, 80000), max_attempts=5, logging_level=10, pydantic_model=Compounds)
asyncio.run(extraction.extract())
end = time.perf_counter()
print(f"cost {end - start} s")