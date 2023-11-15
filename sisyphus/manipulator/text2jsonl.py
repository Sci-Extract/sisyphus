"""
main api: converter
"""
import json

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def chunk_text(text: str, chunk_size) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=15,
    length_function=tiktoken_len
    )
    strip_text = text.replace("\n", " ")
    texts = text_splitter.create_documents([strip_text])
    ret = [text.page_content for text in texts]
    return ret

MODEL = "text-embedding-ada-002"

def render2json(text: list[str], identifier: str, filename: str):
    requests = text
    jobs = [{"model": MODEL, "input": request, "metadata": {"file_name": identifier}} for request in requests]
    
    with open(filename, "a", encoding="utf-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + '\n')

def converter(text: str, metadata:str, jsonl_file_name: str, chunk_size:int = 300) -> None:
    """
    Chunking text and then convert to jsonl with given metadata, noticing that default chunk_size was set to 300.
    """
    text_ls = chunk_text(text, chunk_size)
    render2json(text_ls, metadata, jsonl_file_name)
    