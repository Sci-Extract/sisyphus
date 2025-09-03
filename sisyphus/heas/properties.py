import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Literal


import dspy

from .utils import label_multi_threads
from .embeddings import chroma_db, has_embedded, retrieve, match_subtitles, QUERY_STRENGTH, QUERY_PHASE

class LabelStrength(dspy.Signature):
    """You are an expert in materials science and mechanical testing. Given the following paragraph from a scientific paper on high entropy alloys, determine whether it contains at least one tensile or compressive test property.

    Relevant Properties for Classification:
    A paragraph should be classified as relevant if it contains at least one of the following:

    Yield Strength (YS) (MPa)
    Ultimate Tensile Strength (UTS) (MPa)
    Compressive Strength (MPa)
    Strain (percentage or as a ratio, e.g., true strain or elongation)
    Exclusions:
    Do not classify paragraphs as relevant if they only mention:

    Fracture strength
    Hardness (e.g., Vickers, Brinell, Rockwell)
    Fatigue strength
    Shear strength"""
    paragraph = dspy.InputField()
    relevant: bool = dspy.OutputField(desc='whether relevant')


def label_phase(docs, paragraphs):
    """use vector similar search to first get top 5 paras then using regular expression to filter"""
    source = docs[0].metadata['source']
    res_pattern = re.compile(r'result', re.I)
    res_titles = match_subtitles(docs, res_pattern)
    if source not in has_embedded:
        has_embedded.append(source)
        chroma_db.add_documents(docs)
    similar_docs = retrieve(chroma_db, source, QUERY_PHASE, res_titles, 5) # since this is Document object, we need to find the correspond paragraph object
    para_candidates_vec = []
    for doc in similar_docs:
        for para in paragraphs:
            if para.document == doc:
                para_candidates_vec.append(para) 
    phase_pattern = re.compile(r'\b(FCC|BCC|HCP|L12|B2|Laves|f.c.c.|b.c.c.|h.c.p.|face-centered cubic|body-centered cubic|hexagonal close-packed|intermetallic|IM)\b', re.I)
    for para in para_candidates_vec:
        if phase_pattern.search(para.page_content):
            para.set_types('phase')

def label_strength(docs, paragraphs):
    """use vector similar search to first get top 5 paras then using regular expression to filter"""
    source = docs[0].metadata['source']
    res_pattern = re.compile(r'result', re.I)
    res_titles = match_subtitles(docs, res_pattern)
    if source not in has_embedded:
        has_embedded.append(source)
        chroma_db.add_documents(docs)
    similar_docs = retrieve(chroma_db, source, QUERY_STRENGTH, res_titles, 5) # since this is Document object, we need to find the correspond paragraph object
    para_candidates_vec = []
    for doc in similar_docs:
        for para in paragraphs:
            if para.document == doc:
                para_candidates_vec.append(para)
    mech_pattern = re.compile(r'(\b(MPa|GPa)\b|\d+(\.\d+)?\s*%)')
    para_candidates_reg = [para for para in para_candidates_vec if mech_pattern.search(para.page_content)]
    args = [{'paragraph': para.page_content} for para in para_candidates_reg]
    labeler = dspy.ChainOfThought(LabelStrength)
    results = label_multi_threads(labeler, para_candidates_reg, args, 10)
    for para, result in results:
        if result.relevant:
            para.set_types('strength')

def label_strain_rate(paragraphs):
    pattern = re.compile(r'strain rate', re.I)
    location = re.compile(r'(experiment)|(preparation)|(method)', re.I)
    para_cand = [para for para in paragraphs if location.search(para.metadata['sub_titles'])]
    for para in para_cand:
        if pattern.search(para.page_content):
            para.set_types('strain_rate')

def label_text(docs, paragraphs):
    """label text content of article"""
    label_phase(docs, paragraphs)
    label_strength(docs, paragraphs)
    return paragraphs
