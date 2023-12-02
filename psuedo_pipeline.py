import asyncio
import argparse
import json
import os
import re
import shutil
import time

from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from numpy import array
from openai import OpenAI
from typing import Literal
from termcolor import cprint

from sisyphus.processor.parallel_processor import process_api_requests_from_file
from sisyphus.utils.utilities import ErrorRequestsTracker, Elapsed
from sisyphus.manipulator import create_embedding_jsonl, create_completion_jsonl
from sisyphus.manipulator.df_constructor import build_similarity, select_top_n, construct_df_completion_cls
from pipeline_input import query, system_message, prompt_cls, prompt_sum

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

@Elapsed
def until_all_done(
    error_tracker: ErrorRequestsTracker,
    input_file,
    output_file,
    request_url: Literal["https://api.openai.com/v1/embeddings", "https://api.openai.com/v1/chat/completions"],
    max_requests_per_minute: float,
    max_tokens_per_minute: float
):
    input_path = input_file # copuy
    # get embedding result
    while True:
        output_path = input_path.replace('.jsonl', '_results.jsonl')
        asyncio.run(process_api_requests_from_file(
            requests_filepath=input_path,
            save_filepath=output_path,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=int(3),
            logging_level=int(30), # warning level
        ))
        errors_flag = error_tracker.get_errors_id(output_path)
        if re.search(r'redo', output_path):
            error_tracker.merge_back(output_path, output_file)
        if not errors_flag: # no error detect
            error_tracker.remove_fails(output_file)
            break
        input_path = error_tracker.construct_redo_jsonl(embedding_jsonl) # embedding_jsonl has full record of texts
        cprint("prepare for the failed embedding requests, system dormant time: 10 s\n")
        time.sleep(10)


def pipeline(extract_from: str, query: str, system_message: str, prompt_cls: str, prompt_sum: str):
    # create embedding jsonl
    cprint("Embedding process start...\n",  "green", attrs=["bold"])
    create_embedding_jsonl(extract_from, chunk_size=200) # IC_JA
    cprint("Making requests to openai...\n")

    embedding_error_tracker = ErrorRequestsTracker()
    until_all_done(embedding_error_tracker, embedding_jsonl, embedding_result_jsonl, "https://api.openai.com/v1/embeddings", float(3000), float(1000000))

    # create embedding of paradigm
    cprint("Embedding paradigm sentence start...\n", "green", attrs=["bold"])
    client = OpenAI(api_key=api_key)
    standard_vector = array(client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding)
    cprint("Embedding paradigm sentence finished\n", "red", attrs=["bold"])

    # create similarity
    embedding_df = build_similarity(embedding_result_jsonl, standard_vector, save_file_name="text_similarity.csv")
    selected_df, text_selected_file = select_top_n(embedding_df, top_n=5, save_file_name="text_selected.csv")

    # create completion jsonl (classify)
    cprint("Completion_cls process start...\n", "green", attrs=["bold"])
    create_completion_jsonl(text_selected_file, completion_cls_jsonl, system_message, prompt_cls, 'json')

    completion_cls_error_tracker = ErrorRequestsTracker()
    until_all_done(completion_cls_error_tracker, completion_cls_jsonl, completion_cls_result_jsonl, "https://api.openai.com/v1/chat/completions", float(3500), float(60000))

    # create completion jsonl (summarize)
    cprint("Completion_sum process start...\n", "green", attrs=["bold"])
    df = construct_df_completion_cls(completion_cls_result_jsonl)
    df = df[df['response'] != False]
    df.to_csv(text_filtered, index=False)
    create_completion_jsonl(text_filtered, completion_sum_jsonl, system_message, prompt_sum, 'json', embedding_jsonl)

    completion_sum_error_tracker = ErrorRequestsTracker()
    until_all_done(completion_sum_error_tracker, completion_sum_jsonl, completion_sum_result_jsonl, "https://api.openai.com/v1/chat/completions", float(3500), float(60000))

    return until_all_done.elapsed

# extract_from = "E:\\Projects\\backup\\TEST"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_from", help="The source directory of articles")
    args = parser.parse_args()

    if not os.path.exists("data"):
        os.mkdir("data")
    if len(os.listdir("data")) != 0:
        # move to temp
        current_datetime = datetime.now()
        current_hour = current_datetime.hour
        current_minute = current_datetime.minute
        move_to = os.path.join("temp", f"data_{current_hour}_{current_minute}")
        shutil.move("data", move_to)

        os.mkdir("data") # recreate one

    PROCESS_TIME = pipeline(args.extract_from, query, system_message, prompt_cls, prompt_sum)
    cprint(f"Execution time: {PROCESS_TIME:.2f}s", "cyan", attrs=["bold"])
