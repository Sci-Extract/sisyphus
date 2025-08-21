from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy

from sisyphus.chain.paragraph import Paragraph


class ClassifyCompositionTable(dspy.Signature):
    """Decide whether the table are describing about the chemical composition of HEAs materials
    Guidelines:
    - If the table contains different elements and their atomic/weight percentage, it is a composition table.
    - Be strict about your decisions."""
    table: str = dspy.InputField()
    is_composition: bool = dspy.OutputField()


class LabelTablesStrength(dspy.Signature):
    """You are an expert in materials science and mechanical testing. Given the following CSV table from a scientific paper on high entropy alloys, determine whether it contains at least one tensile or compressive test property.

    Relevant Properties for Classification:
    A table should be classified as relevant if it contains at least one of the following:

    Yield Strength (YS) (MPa)
    Ultimate Tensile Strength (UTS) (MPa)
    Compressive Strength (MPa)
    Strain (percentage or as a ratio, e.g., true strain or elongation)
    Exclusions:
    Do not classify table as relevant if they only mention:

    Fracture strength
    Hardness (e.g., Vickers, Brinell, Rockwell)
    Fatigue strength
    Shear strength"""
    table: str = dspy.InputField()
    contains: bool = dspy.OutputField()


classifier = dspy.ChainOfThought(ClassifyCompositionTable)
table_labeler = dspy.ChainOfThought(LabelTablesStrength)

def label_table(paras: list[Paragraph]):
    """label composition table and strength tables"""
    tables = [para for para in paras if para.is_table()]
    args = [{'table': table.page_content} for table in tables]
    results = label_multi_threads(classifier, tables, args, 5)
    for para, result in results:
        if result.is_composition:
            para.set_types('composition')
    
    # Label strength tables
    properties_results = label_multi_threads(table_labeler, tables, args, 5)
    for para, result in properties_results:
        if result.contains:
            para.set_types('strength')

def label_multi_threads(labeler, paras, args, workers):
    """label the paragraphs in parallel, paras and args should match one by one

    Args:
        labeler: the label function, which should be a dspy compatible caller
        paras: paragraphs
        args: the arguments for the labeler, e.g. [{filed_1: a, field_2: b}, (field_1: c, filed_2: d), ...]
        workers: the number of workers
    """
    with ThreadPoolExecutor(workers) as executor:
        futures = [executor.submit(labeler, **arg) for arg in args]
        future_para = dict(zip(futures, paras))
        para_result_tp = []
        for future in as_completed(futures):
            para_result_tp.append((future_para[future], future.result()))
    return para_result_tp
