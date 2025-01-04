from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import or_

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from sisyphus.utils.helper_functions import get_plain_articledb


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

import re

syn_pattern = re.compile(r'(experiment)|(preparation)', re.I)
def match_subtitles(docs, pattern):
    sub_titles = list(set([doc.metadata["sub_titles"] for doc in docs]))
    target_titles = []
    for title in sub_titles:
        if pattern.search(title):
            target_titles.append(title)
    return target_titles

res_pattern = re.compile(r'result', re.I)
test_pattern = re.compile(r'strain\srate', re.I)
# test_para = [doc for doc in docs if test_pattern.search(doc.page_content)]

QUERY_SYN = """Experimental procedures describing the synthesis and processing of materials, including methods such as melting, casting, rolling, annealing, heat treatment, or other fabrication techniques. Details often include specific temperatures (e.g., °C), durations (e.g., hours, minutes), atmospheric conditions (e.g., argon, vacuum), mechanical deformation (e.g., rolling reduction), and microstructural characterization steps. Mentions of material compositions, purity levels, and equipment used are common indicators."""
QUERY_MECHANICAL = """Mechanical properties of high entropy alloys, stress-strain curves, yield strength, ultimate tensile strength, tensile strain, elongation, alloy composition, alloying effects on strength, ductility, engineering stress-strain behavior."""
QUERY_PHASE = """Phase characterization of high entropy alloys, microstructure analysis, crystal structures, phase transitions, XRD patterns, lattice parameters, grain morphology, recrystallization, secondary phases, alloying effects on phases, defect structures, and phase stability."""
K = 3

db = get_plain_articledb('300_heas')
def test_one(article):
    docs = db.get(article)
    syn_titles, res_titles = list(map(lambda a,b: a or b, map(partial(match_subtitles, docs), [syn_pattern, res_pattern]), [None, None]))
    with ThreadPoolExecutor(max_workers=20) as worker:
        iterable = [[item] for item in docs]
        worker.map(vector_store.add_documents, iterable)
    def retrieve(vector_store, query, sub_titles):
        return vector_store.similarity_search(
            query,
            k=K,
            filter={
                "sub_titles": {
                    "$in": sub_titles
                }
            }
        )
    syn_paras = retrieve(vector_store, QUERY_SYN, syn_titles)
    mechanical_paras = retrieve(vector_store, QUERY_MECHANICAL, res_titles)
    phase_paras = retrieve(vector_store, QUERY_PHASE, res_titles)
    test_paras = [doc for doc in docs if test_pattern.search(doc.page_content)]

    doi = docs[0].metadata['doi']
    with open('inspect_retriever.txt', encoding='utf8', mode='a') as f:
        f.write(f'doi: {doi}\n\n')
        f.write('experimental\n')
        for p in syn_paras:
            f.write(p.page_content + '\n')
        f.write('\n')
        f.write('mechanical\n')
        for p in mechanical_paras:
            f.write(p.page_content + '\n')
        f.write('\n')
        f.write('phase\n')
        for p in phase_paras:
            f.write(p.page_content + '\n')
        f.write('\n')
        f.write('test parameter\n')
        for p in test_paras:
            f.write(p.page_content + '\n')
        f.write('\n')
        f.write('====================')

import os
files = os.listdir('articles_processed')[:10]
with ThreadPoolExecutor(max_workers=10) as worker:
    worker.map(test_one, files)
