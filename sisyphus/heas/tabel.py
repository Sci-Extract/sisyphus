from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy

from .paragraph import Paragraph


class ClassifyCompositionTable(dspy.Signature):
    """Decide whether the table are describing about the chemical composition of HEAs materials
    Guidelines:
    - If the table contains different elements and their atomic/weight percentage, it is a composition table.
    - Be strict about your decisions."""
    table: str = dspy.InputField()
    is_composition: bool = dspy.OutputField()


class LabelTablesProperties(dspy.Signature):
    """From the provided CSV table, generate a Python list of item categories that have data present. The categories to check are:

    1. strength (valid ONLY if at least one of these properties exists: YS, UTS, or strain)
    2. youngs_modulus (elastic modulus of materials)
    3. hardness (material hardness measurements)
    4. grain_size (material grain size measurements, excluding lattice parameters)

    Requirements:
    - Include a category in the output list ONLY if relevant data for that category exists in the CSV
    - Return an empty list if any required category is completely missing data
    - Do not include categories with no data
    - For strength to be valid, the CSV must contain at least one of: YS, UTS, or strain
    - Shear stress and corrosion resistance are NOT valid strength properties
    - Lattice parameters are NOT valid grain_size properties

    Example output: ['strength', 'hardness'] or [] if missing data
    """
    table: str = dspy.InputField()
    properties: list[Literal['strength', 'youngs_modulus', 'hardness', 'grain_size']] = dspy.OutputField(desc='if none of the above is present, return an empty list')   


classifier = dspy.ChainOfThought(ClassifyCompositionTable)
table_labeler = dspy.ChainOfThought(LabelTablesProperties)

def label_table(paras: list[Paragraph]):
    tables = [para for para in paras if para.is_table()]
    args = [{'table': table.page_content} for table in tables]
    results = label_multi_threads(classifier, tables, args, 5)
    for para, result in results:
        if result.is_composition:
            para.set_types('composition')
    properties_results = label_multi_threads(table_labeler, tables, args, 5)
    for para, result in properties_results:
        para.set_types(result.properties)

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
            
