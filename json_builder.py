"""
this code is not elegant, needed refactoring...
"""

import os

from sisyphus.reader.text_chunking import chunk_text
from sisyphus.reader.render import render2json

# with open('ic2c00623.txt', encoding='utf-8') as f:
#     article = f.read()

# text_chunks = chunk_text(article)
# print(len(text_chunks))

# identifier = "ic2c00623"
# render2json(text_chunks, identifier, "out.json")

os.chdir("sisyphus\\data\\ICarticles")

dir_ls = os.listdir()
for dir in dir_ls:

    os.chdir(dir)

    files = os.listdir()
    txt_files = [file for file in files if file.endswith('.txt')]

    for txt in txt_files:
        with open(txt, encoding='utf-8') as f:
            article = f.read()
        text_chunks = chunk_text(article)
        identifier = dir
        os.chdir("E:\Projects\sisyphus")
        render2json(text_chunks, identifier, "embedding_text.jsonl")
    
    os.chdir("sisyphus\\data\\ICarticles")
