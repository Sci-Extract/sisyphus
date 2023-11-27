"""
in and out, that's the essence of what this package does.
"""
import os
import json
from typing import Literal

import pandas as pd

from .jsonl_constructor import converter_embedding, get_target_dir_txt, completion_json_formatter, write_jsonl, embedding_json_formatter


def get_text_by_id(task_id: list[int], file_path: str, key="input") -> str:
    json_list = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line)
            json_list.append(line_json)
    for id in task_id:
        for unit in json_list:
            if id == unit["metadata"]["task_id"]:
                yield unit[key]

def get_write_mode(file_path):
    write_mode = 'a'
    if os.path.exists(file_path):
        user_input = input(f"File already exists [{file_path}], over-write ('w') or keep addding ('a'): ")
        if user_input == 'w' or 'W':
            write_mode = 'w'
        elif user_input == 'a' or 'A': # maybe useful when batch process.
            write_mode = 'a'
    return write_mode

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

def create_embedding_jsonl(source: str):
    """source_path: the paraent folder of articles. 
    article_dirs: target articles directory names.
    * suggested file structure:
    articles/
        <article_name_1>
            <article_name_1>.txt
            <article_name_1>.xml/html
            <SI>
        <article_name_2>
        ... 
    output_dir: location where your output jsonl files store.
    output_josnl_name: name of the jsonl file.
    """

    # defaults
    jsonl_file_name="embedding.jsonl"
    jsonl_file_dir="data"
    file_path = os.path.join(jsonl_file_dir, jsonl_file_name)
    write_mode = get_write_mode(file_path)
    if write_mode == 'w':
        open(file_path, 'w', encoding='utf-8').close() # clear the content

    task_id_generator = task_id_generator_function()
    for article in os.listdir(source):
        article_path = os.path.join(source, article)
        content = get_target_dir_txt(article_path)
        converter_embedding(content, file_name=article, task_id_generator=task_id_generator, write_mode='a', jsonl_file_name=jsonl_file_name, jsonl_file_dir=jsonl_file_dir, chunk_size=300) # modify the name if needed
    
def create_completion_jsonl(source: str, file_path: str, system_message: str, prompt: str, required_format: Literal["json", "text"], text_jsonl: str = None, model="gpt-3.5-turbo-1106", temperature: float = 0.0):
    if required_format == "json":
        response_format={"type": "json_object"}
    elif required_format == "text":
        response_format={"type": "text"}
    else:
        raise ValueError
    
    jobs = []
    df = pd.read_csv(source)
    if text_jsonl:
        gen_text = get_text_by_id(df["task_id"].tolist(), text_jsonl)

    for _, row in df.iterrows():
        if text_jsonl:
            text = next(gen_text)
        else:
            text = row["content"]
        article_name = row["file_name"]
        task_id = row["task_id"]
        
        user_message = prompt + '```' + text + '```'
        jobs.append(completion_json_formatter(system_message, user_message, article_name, task_id, model, temperature, response_format))
    
    write_mode = get_write_mode(file_path)
    if write_mode == 'w':
        open(file_path, 'w', encoding='utf-8').close() # clear the content
    write_jsonl(file_path, jobs, write_mode='a')
