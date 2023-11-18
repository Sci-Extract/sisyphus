"""
in and out, that's the essence of what this package does.
"""
import os

from .text2jsonl import converter


def get_target_dir_txt(dir_path: str) -> str:
    files = os.listdir(dir_path)
    txt_file = [file for file in files if file.endswith('.txt')][0] # normally there is only one txt file, if there is plural number, modify the code.
    file_path = os.path.join(dir_path, txt_file)
    with open(file_path, encoding='utf-8') as file:
        content = file.read()
        return content

def pipeline_embedding_construct(source_path: str, article_dirs: list[str], output_dir: str, output_jsonl_name: str):
    """source_path: the path where the articles located. article_dirs: target articles dirnames. output_dir: location where your output jsonl files store.
    output_josnl_name: name of the jsonl file.
    """
    for article_dir in article_dirs:
        dir_path = os.path.join(source_path, article_dir)
        content = get_target_dir_txt(dir_path)
        converter(content, metadata=article_dir, jsonl_file_dir=output_dir, jsonl_file_name=output_jsonl_name) # the name of article directory is usually the name of the article identity
