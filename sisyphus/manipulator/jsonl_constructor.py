"""
main api: converter
"""
import json
import os
import re
from typing import Generator

import tiktoken


tokenizer = tiktoken.get_encoding("cl100k_base")
sci_notation = ["ca.", "calc.", "cal.", "no.", "e.g.", "i.e."]
sci_notation_pattern = r'(ca\.)|(calc\.)|(cal\.)|(no\.)|(e\.g\.)|(i\.e\.)'

def detect_sci_dot(after_dot: str, before_dot: str):
    if re.match(r"\w", after_dot):
        return True
    elif re.search(sci_notation_pattern, before_dot, re.I):
        return True
    else:
        return False

    
# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n):
    """Returns successive n-sized chunks from provided text."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            dot_of_science = False if j == len(tokens) else detect_sci_dot(tokenizer.decode(tokens[j:j+1]), tokenizer.decode(tokens[j-4: j])) # judge whether the dot is the regular one. return False if it is. -4: comply with sci_notation_pattern.
            if chunk.endswith('.') and not dot_of_science:
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def get_target_dir_txt(dir_path: str) -> str:
    files = os.listdir(dir_path)
    txt_file = [file for file in files if file.endswith('.txt')][0] # normally there is only one txt file, if there is plural number, modify the code.
    file_path = os.path.join(dir_path, txt_file)
    with open(file_path, encoding='utf-8') as file:
        content = file.read()
        return content

def embedding_json_formatter(text: list[str], identifier: str, file_dir: str, file_name: str, task_id_generator: Generator | None, write_mode):
    model = "text-embedding-3-large"
    requests = text
    if bool(task_id_generator):
        jobs = [{"model": model, "input": request, "metadata": {"file_name": identifier, "task_id": next(task_id_generator)}} for request in requests]
    else:
        jobs = [{"model": model, "input": request, "metadata": {"file_name": identifier, "task_id": 0}} for request in requests]
    file_path = os.path.join(file_dir, file_name)
    write_jsonl(file_path, jobs, write_mode)

def write_jsonl(file_path, jobs, write_mode):
    
    with open(file_path, write_mode, encoding="utf-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + '\n')

def converter_embedding(text: str, file_name:str, jsonl_file_name: str, task_id_generator: Generator, write_mode, jsonl_file_dir: str ="data", chunk_size: int = 300) -> None:
    """
    Chunking text and then convert to jsonl with given metadata, noticing that default chunk_size was set to 300.
    """
    generator = create_chunks(text, chunk_size)
    text_ls = [dec.strip() for g in generator if "https://api.elsevier.com/" not in (dec:=tokenizer.decode(g))]
    embedding_json_formatter(text_ls, file_name, jsonl_file_dir, jsonl_file_name, task_id_generator, write_mode)

def create_completion_from_embedding(embedding_file: str, out_file: str, system_message: str, prompt: str):
    with open(embedding_file, 'r', encoding='utf-8') as f_out:
        jobs = []
        for line in f_out:
            content = json.loads(line)
            metadata = content["metadata"]
            doc = content["input"]
            user_message = prompt + '```' + doc + '```'
            jobs.append(completion_json_formatter(system_message, user_message, metadata["file_name"], metadata["task_id"]))
    with open(out_file, 'w', encoding='utf-8') as fp:
        for job in jobs:
            fp.write(json.dumps(job))
            fp.write("\n")


def completion_json_formatter(system_message: str, user_message: str, article_name, task_id, model="gpt-3.5-turbo-0125", temperature: float = 0.0, response_format={"type": "json_object"}):
    json_format = {
    "messages": [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ],
    "model": model,
    "temperature": temperature,
    "response_format": response_format,
    "metadata": {"file_name": article_name, "task_id": task_id}
}
    return json_format

def completion_json_formatter_with_doc(doc:str, system_message: str, user_message: str, article_name, task_id, model="gpt-3.5-turbo-0125", temperature: float = 0.0, response_format={"type": "json_object"}):
    json_format = {
    "messages": [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ],
    "model": model,
    "temperature": temperature,
    "response_format": response_format,
    "metadata": {"file_name": article_name, "task_id": task_id, "doc": doc}
}
    return json_format