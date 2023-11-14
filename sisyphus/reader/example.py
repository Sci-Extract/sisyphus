"""
for illustration, depend on use case.
"""

import os

from sisyphus.reader.text_chunking import chunk_text
from sisyphus.reader.render import render2json

# for single text
text = "Description of the properties of NLO materials, include second harmonic generation (SHG), band gaps (Eg), birefringence, phase match, absorption edge, laser induced damage thersholds (LIDT). reports values unit such as (eV, pm/V, MW/cm2, nm), and the SHG value is given in multiples of KDP or AgGaS2."

text_chunks = chunk_text(text)
print(text_chunks)

identifier = "paradigm"
render2json(text_chunks, identifier, "paradigm.jsonl")

# for multiple
# os.chdir("sisyphus\\data\\ICarticles")

# dir_ls = os.listdir()
# for dir in dir_ls:

#     os.chdir(dir)

#     files = os.listdir()
#     txt_files = [file for file in files if file.endswith('.txt')]

#     for txt in txt_files:
#         with open(txt, encoding='utf-8') as f:
#             article = f.read()
#         text_chunks = chunk_text(article)
#         identifier = dir
#         os.chdir("E:\Projects\sisyphus")
#         render2json(text_chunks, identifier, "embedding_text.jsonl")
    
#     os.chdir("sisyphus\\data\\ICarticles")
