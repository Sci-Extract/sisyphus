import re

import dspy
from langchain_core.documents import Document


from sisyphus.chain.paragraph import Paragraph
from .synthesis import label_syn_paras
from .properties import label_text, label_strain_rate, label_grain_size
from .tabel import label_table


def label_properties_restricted(docs, paras: list[Paragraph]):
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
    label_text(docs, restricted_paras)

def label_paras(docs: list[Document]):
    """label paragraphs for high entropy alloys paper"""
    paras = [Paragraph(doc, id_) for id_, doc in enumerate(docs)]

    label_syn_paras(docs, paras) # label synthesis paragraphs
    label_table(paras) # label chemical composition, strength, processing parameters, grain size tables.
    label_properties_restricted(docs, paras) # label phase and strength texts
    label_strain_rate(paras)
    label_grain_size(paras)

    return paras

def label_only_syn_paras(docs: list[Document]):
    """label synthesis paragraphs only"""
    paras = [Paragraph(doc, id_) for id_, doc in enumerate(docs)]
    with dspy.context(lm=dspy.LM('openai/gpt-4.1-mini')):
        label_syn_paras(docs, paras) # label synthesis paragraphs
    return paras
