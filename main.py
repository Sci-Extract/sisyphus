import asyncio
import time

from dotenv import find_dotenv, load_dotenv

from sisyphus.processor.llm_extraction import Extraction


_ = load_dotenv(find_dotenv())

from_ = "data_articles"
system_message = "You are reading a piece of text from chemistry articles about nonlinear optical (nlo) materials and you are required to response based on the context provided by the user."
prompt_cls = \
"""Given a text quoted by triple backticks, judge whether the text contains the desired information. Return a JSON object with the following criteria:

a. Check if the text includes at least one chemical compound (e.g., KBBF, BaB2O4, abbreviation, or pronoun).
b. Verify if the text includes at least one nonlinear optical (nlo) materials property corresponding to a specific chemical compound, such as second harmonic generation coefficient (dij), band gaps (Eg), birefringence, absorption edge (cutoff edge), or LIDT.
c. Confirm that the text contains at least one numerical value (e.g. 4.5 eV, 0.45 pm V-1) corresponding to a nonlinear optical (nlo) materials property.

Sequentially examine each criterion. If any criterion is not met, return False for that criterion. Output a JSON complying with the schema:

{
  "a": true/false,
  "b": true/false,
  "c": true/false
}

"""
prompt_sum = \
"""Given a text quoted by triple backticks, I want to extract information about nonlinear optical (nlo) compounds from a text.
The information should include the name of the compound and numerical values of properties such as the second harmonic generation coefficient dij (shg). Note that shg sometimes given in multiple value corresponds to different dij, record every of them. band gaps (Eg), birefringence, absorption edge (cutoff edge), or LIDT (laser damage threshold). 
If shg value is given by the times of a standard material, set the unit to the standard material (e.g., "unit": "KDP").
The compound name should be in properly formulated chemical format. If a value is given in relation to a standard material, the material should be the unit. If any property is not documented, fill it as null.
Extract the values as such in the below JSON format:

{
  "compounds":[
    {
      "name":,
      "shg": [{"value":, "unit":, "dij":}],
      "eg": {"value":, "unit":},
      "birefringence": {"value":, "unit":},
      "cutoff": {"value":, "unit":},
      "lidt": {"value:, "unit":}
    },
    // Repeat the same structure for each nlo compound
  ]
}

"""

query = "Description of the properties of NLO materials, include the name of nlo material (e.g. KBBF, Na4B8O9F10), second harmonic generation SHG (e.g. 0.8 pm/V, 3 × KDP), band gaps Eg (e.g. 6.2 eV), birefringence, phase match, absorption edge, laser induced damage thersholds (LIDT). reports values unit such as (eV, pm/V, MW/cm2, nm), and the SHG value is sometimes given in multiples of KDP or AgGaS2."

d = dict(query=query, prompt_cls=prompt_cls, prompt_sum=prompt_sum, system_message=system_message)

from pydantic_model_nlo import Model
start = time.perf_counter()
extraction = Extraction(from_=from_, save_filepath="test_results_debug.jsonl", query_and_prompts=d, embedding_limit=(5000, 1000000), completion_limit=(5000, 80000), max_attempts=5, logging_level=10)
asyncio.run(extraction.extract(sample_size=1, pydantic_model=Model))
end = time.perf_counter()
print(f"cost {end - start} s")