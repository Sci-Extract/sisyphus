"""
in and out, that's the essence of what this package does.
"""
import os
import json
import uuid
from typing import Literal, Tuple

import pandas as pd
from openai import OpenAI
from chromadb.api.models import Collection

from .jsonl_constructor import converter_embedding, get_target_dir_txt, completion_json_formatter, write_jsonl, embedding_json_formatter, completion_json_formatter_with_doc


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

def check_content(file_path):
    """return False if empty, if file not existed, create one and return False"""
    if not os.path.exists(file_path):
        open(file_path, 'w').close()
        return False
    handler = open(file_path, encoding='utf8')
    lines = handler.readlines()
    handler.close()
    return bool(lines)

def create_embedding_jsonl(source: str, duplicated_articles: list[str], chunk_size: int = 300, sample_size: int = None) -> Tuple[str, bool]:
    """retrun the create file path, 
    source_path: the paraent folder of articles. 
    article_dirs: target articles directory names.
    * suggested file structure:
    source/
        publisher1/
            article_name_1
                article_name_1.txt
                article_name_1.xml/html
                SI
            article_name_2
        publisher2/
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
    
    count = 0

    task_id_generator = task_id_generator_function()
    for publisher in os.listdir(source):
        publisher_dir = os.path.join(source, publisher)
        for article in os.listdir(publisher_dir):
            if sample_size:
                if count == sample_size:
                    return file_path, check_content(file_path=file_path)
                count += 1
            if article in duplicated_articles: # skip duplicated articles
                continue
            article_path = os.path.join(publisher_dir, article)
            content = get_target_dir_txt(article_path)
            if content: # sometimes failed to get RSC articles, it's empty text.
                converter_embedding(content, file_name=article, task_id_generator=task_id_generator, write_mode='a', jsonl_file_name=jsonl_file_name, jsonl_file_dir=jsonl_file_dir, chunk_size=chunk_size) # modify the name if needed
            
    return file_path, check_content(file_path=file_path)
    
def create_completion_jsonl(df, file_path: str, system_message: str, prompt: str, required_format: Literal["json", "text"], text_jsonl: str = None, model="gpt-3.5-turbo-1106", temperature: float = 0.0):
    """old implementation of creating completion jsonl"""
    if required_format == "json":
        response_format={"type": "json_object"}
    elif required_format == "text":
        response_format={"type": "text"}
    else:
        raise ValueError
    
    jobs = []
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

# Get the name of articles, then get the duplicate ones
def get_running_names(directory="data_articles", sample_size=None):
    names = []
    count = 0
    for publisher in os.listdir(directory):
        publisher_dir = os.path.join(directory, publisher)
        for article in os.listdir(publisher_dir):
            names.append(article)
            if sample_size:
                count += 1
                if count == sample_size:
                    return names
    return names

def get_duplicated_names(running_names, chroma_collection: Collection):
    duplicated_names = []
    for name in running_names:
        res = chroma_collection.get(
            where={"file_name": name},
            limit=1
        )
        if res["ids"]:
            duplicated_names.append(name)
    return duplicated_names

# add new embeddings to db
def add_embeddings(chroma_collection: Collection, result_file="data\\embedding_results.jsonl"):
    with open(result_file, encoding='utf8') as file:
        for line in file:
            embedding_element = json.loads(line)
            embedding_element[2].pop("task_id")
            metadata = embedding_element[2]
            embedding = embedding_element[1]["embedding"]
            document = embedding_element[0]["input"]
            chroma_collection.add(
                ids=[str(uuid.uuid1())],
                embeddings=embedding,
                metadatas=metadata,
                documents=document
            )

# get the top 5 relevant chunks and construct completion jsonl
def fetch_and_construct(chroma_collection: Collection, search_query: str, running_names: list[str], system_message: str, prompt: str):
    """Use query to fetch top 5 relevant chunks from chromadb, then use these chunks to construct jsonl for later use.

    Args:
        chroma_collection: The name of the chroma database.
        search_query: The query used for searching.
        running_names: The file names list of runnning tasks
        system_message: prompt system message.
        prompt: prompt itself.

    """
    id_generator = task_id_generator_function()
    save_filepath = os.path.join("data", "completion_cls.jsonl")

    def construct(docs: list[str], article_name: str, save_filepath: str, system_message: str, prompt:str, model="gpt-3.5-turbo-1106"):
        jobs = []
        for doc in docs:
            user_message = prompt + '```' + doc + '```'
            jobs.append(completion_json_formatter_with_doc(doc, system_message, user_message, article_name, next(id_generator), model))
        write_jsonl(save_filepath, jobs, write_mode='a')

    with OpenAI() as client:
        embedding_query = client.embeddings.create(
            input=search_query,
            model="text-embedding-ada-002")
        query_vector = embedding_query.data[0].embedding
    for name in running_names:
        results = chroma_collection.query(
        query_embeddings=query_vector,
        n_results=10,
        where={"file_name": name}
        )
        docs = results["documents"][0]
        construct(docs=docs, article_name=name, save_filepath=save_filepath, system_message=system_message, prompt=prompt)
    return save_filepath


def get_candidates_construct(from_file: str, system_message: str, prompt: str):
    """Get candidates from re-checking results file, construct final jsonl for extraction from it.
    """
    with open(from_file, encoding='utf8') as file:
        jobs = []
        for line in file:
            result = json.loads(line)
            if result[1] == "Failed":
                continue
            content = result[1]["content"]
            values = content.values()
            if all(values):
                doc = result[2]["doc"]
                user_message = prompt + '```' + doc + '```'
                jobs.append(completion_json_formatter(
                    system_message=system_message,
                    user_message=user_message,
                    article_name=result[2]["file_name"],
                    task_id=result[2]["task_id"]
                ))
    save_filepath = os.path.join("data", "completion_sum.jsonl")
    write_jsonl(save_filepath, jobs, write_mode='a')
    return save_filepath