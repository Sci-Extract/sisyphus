import asyncio
import json
import os
import re
import shutil
import time

from dotenv import load_dotenv, find_dotenv
from numpy import array
from openai import OpenAI
from termcolor import colored, cprint

from sisyphus.processor.parallel_processor import process_api_requests_from_file
from sisyphus.utils.utilities import ErrorRequestsTracker
from sisyphus.manipulator import create_embedding_jsonl, create_completion_jsonl
from sisyphus.manipulator.df_constructor import build_similarity, select_top_n, construct_df_completion_cls

# load environment
_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

# embedding and completion api
embedding_url = "https://api.openai.com/v1/embeddings"
completion_url = "https://api.openai.com/v1/chat/completions"

# default file path /* Attention: file was generated during the process */
embedding_jsonl = os.path.join("data", "embedding.jsonl")
embedding_result_jsonl = embedding_jsonl.replace('.jsonl', '_results.jsonl')

paradigm_jsonl = os.path.join("data", "paradigm.jsonl")
paradigm_result_jsonl = paradigm_jsonl.replace('.jsonl', '_results.jsonl')

completion_cls_jsonl = os.path.join("data", "completion_cls.jsonl")
completion_cls_result_jsonl = completion_cls_jsonl.replace('.jsonl', '_results.jsonl')

completion_sum_jsonl = os.path.join('data', 'completion_sum.jsonl')
completion_sum_result_jsonl = completion_sum_jsonl.replace('.jsonl', '_results.jsonl')

text_filtered = os.path.join('data', "text_filtered.csv")



# create embedding jsonl
cprint("Embedding process start...",  "red", attrs=["bold"])
create_embedding_jsonl("E:\\Projects\\backup\\IC_JA")
cprint("Making requests to openai...")

# check exsitence
if os.path.exists(embedding_jsonl):
    pass
else:
    raise FileNotFoundError

if os.path.exists(embedding_result_jsonl):
    cprint("result file already exists, move to temp file", on_color="on_cyan")
    file_name = os.path.basename(embedding_result_jsonl)
    temp_file_path = os.path.join("temp", file_name)
    number = 1
    while True:
        if os.path.exists(temp_file_path):
            temp_file_path = temp_file_path.replace('.jsonl', f'_{number}.jsonl')
            number += 1
        else:
            break
    shutil.move(embedding_result_jsonl, temp_file_path)

start = time.perf_counter()
embedding_error_tracker = ErrorRequestsTracker()
input_path = embedding_jsonl # for reference

# get embedding result
while True:
    output_path = input_path.replace('.jsonl', '_results.jsonl')
    asyncio.run(process_api_requests_from_file(
        requests_filepath=input_path,
        save_filepath=output_path,
        request_url=embedding_url,
        api_key=api_key,
        max_requests_per_minute=float(3000*0.5),
        max_tokens_per_minute=float(1000000*0.5),
        token_encoding_name="cl100k_base",
        max_attempts=int(3),
        logging_level=int(30), # warning level
    ))
    errors_flag = embedding_error_tracker.get_errors_id(output_path)
    if re.search(r'redo', output_path):
        embedding_error_tracker.merge_back(output_path, embedding_result_jsonl)
    if not errors_flag: # no error detect
        embedding_error_tracker.remove_fails(embedding_result_jsonl)
        break
    input_path = embedding_error_tracker.construct_redo_jsonl(embedding_jsonl)
    cprint("prepare for the failed embedding requests, system dormant time: 10 s")
    time.sleep(10)

end = time.perf_counter()
cprint(f"Embedding process finished, Runtime: {end - start:.2f}", "red", attrs=["bold"])




# create embedding of paradigm
cprint("Start embedding the paradigm sentence", "red")
query = "Description of the properties of NLO materials, include the name of nlo material (e.g. KBBF, Na4B8O9F10), second harmonic generation SHG (e.g. 0.8 pm/V, 3 × KDP), band gaps Eg (e.g. 6.2 eV), birefringence, phase match, absorption edge, laser induced damage thersholds (LIDT). reports values unit such as (eV, pm/V, MW/cm2, nm), and the SHG value is sometimes given in multiples of KDP or AgGaS2."
client = OpenAI(api_key=api_key)
standard_vector = array(client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding)
cprint("Finish embedding the paradigm sentence", "red")

# create similarity
embedding_df = build_similarity(embedding_result_jsonl, standard_vector, save_file_name="text_similarity.csv")
selected_df, text_selected_file = select_top_n(embedding_df, top_n=5, save_file_name="text_selected.csv")




# create completion jsonl (classify)
cprint("Completion process start...", "red")
system_message = "You are reading a piece of text from chemistry articles about nonlinear optical (nlo) materials and you are required to response based on the context provided by the user."
prompt_sum = \
"""The text is quoted by triple backticks.
nlo properties: second harmonic generation coefficient (dij), band gaps (Eg), birefringence, absorption edge (cutoff edge), or LIDT
Summarize the text into json format, only include nlo compound which has at least one property, the json schema should comply with:
{
    "compound_name": <name>,
    "shg": {"value": <value>, "unit": <unit>},
    "eg": {"value": <value>, "unit": <unit>},
    "birefringence": {<value>, "unit": <unit>},
    "lidt": {<value>, "unit": <unit>}
}
if find multiple nlo compounds, the response should be a list of json.
if the value is given by the times of standard material, then the unit is set to the standard material (e.g. "unit": "KDP")
Filled with null if any field not find.

"""
prompt_cls = \
"""The text is quoted by triple backticks.

Judge whether the text has the desired information, Return True if all criteria listed below match, False if any do not.
a. Includes at least one chemical compound (e.g. KBBF, BaB2O4, abbreviation or pronoun).
b. Includes at least one nonlinear optical (nlo) materials property correspond to the specific chemical compound, such as second harmonic generation coefficient (dij), band gaps (Eg), birefringence, absorption edge (cutoff edge), or LIDT.
c. Contains at least one numerical value corresponding to a nonlinear optical (nlo) materials property.

Sequentially examine each criteria.
If any criterion is not met, return False.
output json comply with schema:
{
  "a": true/false,
  "b": true/false,
  "c": true/false
}

"""
create_completion_jsonl(text_selected_file, completion_cls_jsonl, system_message, prompt_cls, 'json')

start = time.perf_counter()
completion_cls_error_tracker = ErrorRequestsTracker()
input_path = completion_cls_jsonl # for reference

# get embedding result
while True:
    output_path = input_path.replace('.jsonl', '_results.jsonl')
    asyncio.run(process_api_requests_from_file(
        requests_filepath=input_path,
        save_filepath=output_path,
        request_url=completion_url,
        api_key=api_key,
        max_requests_per_minute=float(3500*0.5),
        max_tokens_per_minute=float(90000),
        token_encoding_name="cl100k_base",
        max_attempts=int(3),
        logging_level=int(30), # warning level
    ))
    errors_flag = completion_cls_error_tracker.get_errors_id(output_path)
    if re.search(r'redo', output_path):
        completion_cls_error_tracker.merge_back(output_path, completion_cls_result_jsonl)
    if not errors_flag: # no error detect
        embedding_error_tracker.remove_fails(completion_cls_result_jsonl)
        break
    input_path = completion_cls_error_tracker.construct_redo_jsonl(completion_cls_jsonl)
    cprint("prepare for the failed completion requests, system dormant time: 10 s")
    time.sleep(10)

end = time.perf_counter()
cprint(f"Completion classify process finished, Runtime: {end - start:.2f}s", "red", attrs=["bold"])




# create completion jsonl (summarize)
cprint("Completion process start...", "red")
df = construct_df_completion_cls(completion_cls_result_jsonl)
df = df[df['response'] != False]
df.to_csv(text_filtered, index=False)
create_completion_jsonl(text_filtered, completion_sum_jsonl, system_message, prompt_sum, 'json', embedding_jsonl)

start = time.perf_counter()
completion_sum_error_tracker = ErrorRequestsTracker()
input_path = "data\\completion_sum_redo.jsonl" # for reference

# get embedding result
while True:
    output_path = input_path.replace('.jsonl', '_results.jsonl')
    asyncio.run(process_api_requests_from_file(
        requests_filepath=input_path,
        save_filepath=output_path,
        request_url=completion_url,
        api_key=api_key,
        max_requests_per_minute=float(3500*0.5),
        max_tokens_per_minute=float(90000),
        token_encoding_name="cl100k_base",
        max_attempts=int(3),
        logging_level=int(30), # warning level
    ))
    errors_flag = completion_sum_error_tracker.get_errors_id(output_path)
    if re.search(r'redo', output_path):
        completion_sum_error_tracker.merge_back(output_path, completion_sum_result_jsonl)
    if not errors_flag: # no error detect
        embedding_error_tracker.remove_fails(completion_sum_result_jsonl)
        break
    input_path = completion_sum_error_tracker.construct_redo_jsonl(completion_sum_jsonl)
    cprint("prepare for the failed completion requests, system dormant time: 10 s")
    time.sleep(10)

end = time.perf_counter()
cprint(f"Completion summary process finished, Runtime: {end - start:.2f}", "red", attrs=["bold"])
