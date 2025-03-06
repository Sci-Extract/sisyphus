import re

import dspy
from langchain_core.documents import Document

from .paragraph import Paragraph
from .synthesis import label_syn_paras
from .properties import label_properties
from .tabel import label_table


def label_properties_restricted(paras: list[Paragraph]):
    """do not label content in introduction, conflict, acknowledge, support, and synthesis and table section within paper"""
    intro_pattern = re.compile(r'(introduction)', re.I)
    syn_pattern = re.compile(r'(experiment)|(preparation)|(method)', re.I)
    conflict_pattern = re.compile(r'conflict', re.I)
    acknowledge_pattern = re.compile(r'acknowledge', re.I)
    support_pattern = re.compile(r'support', re.I)
    
    restricted_paras = []
    for para in paras:
        is_irrelevant = any(pattern.search(para.metadata['sub_titles']) for pattern in [intro_pattern, syn_pattern, conflict_pattern, acknowledge_pattern, support_pattern])
        if para.is_table() or is_irrelevant:
            continue
        restricted_paras.append(para)
    label_properties(restricted_paras)

def label_paras(docs: list[Document]):
    """label paragraphs for high entropy alloys paper"""
    paras = [Paragraph(doc, id_) for id_, doc in enumerate(docs)]

    with dspy.context(lm=dspy.LM('openai/gpt-4o-mini')):
        label_syn_paras(docs, paras)
        label_table(paras)
        label_properties_restricted(paras)

    return paras

def label_paras_test(docs):
    """label paragraphs for high entropy alloys paper, the LM are the same as caller module"""
    paras = [Paragraph(doc, id_) for id_, doc in enumerate(docs)]

    label_syn_paras(docs, paras)
    label_properties_restricted(paras)

    return paras

def label_syn_only(docs):
    paras = [Paragraph(doc, id_) for id_, doc in enumerate(docs)]
    label_syn_paras(docs, paras)

    return paras
