import re

import dspy

from sisyphus.utils.helper_functions import render_docs, get_title_abs, reorder_docs

class GetAcronyms(dspy.Signature):
    """Find acronyms or labels from the given chemical formulas, if you can't find any, return an empty list
    Note:
    - Acronyms must either: Consist only of letters and digits (e.g., "2B", "1C"); Be in all capital letters (e.g., "AGS").
    - If the chemical mention is considered as an acronym if and only if itself is an acronym, not the prefix or suffix.
    - The extracted acronyms must match exactly as they appear in the input, without any modifications."""
    cems: list[str] = dspy.InputField()
    acronyms: list[str] = dspy.OutputField(desc="list of acronyms or labels found in cems e.g. ['2b', 'LFP']")

get_acronym_ex = [dspy.Example(cems=['LiB6O9F', 'LBO', '1a', '(BO2F2)3-'], acronyms=['LBO', '1a']).with_inputs('cems'),
                  dspy.Example(cems=['Pnma-MgSiO3', 'β-BaB2O4'], acronyms=[]).with_inputs('cems')]
get_acronym_2_shots = dspy.LabeledFewShot().compile(dspy.ChainOfThought(GetAcronyms), trainset=get_acronym_ex)

class DropInvalid(dspy.Signature):
    """Drop chemical formulas that are not valid based on user defined rules"""
    cems: list[str] = dspy.InputField()
    dropped: list[str] = dspy.OutputField()

class Standardization(dspy.Signature):
    """Standardize chemical formulas to remove any redundant information to only keep the main chemical formula."""
    cems: list[str] = dspy.InputField()
    standardized_cems: dict = dspy.OutputField(desc='a dict which key is the original cem and value is the standardized cem, e.g. {"LiB6O9F": "LiB6O9F"}')

class ProcessChemicalNames(dspy.Signature):
    """Find correspond chemical formula for given acronyms or labels using context information.
    Note:
    - You should find every chemical formula for a given acronym or label, if possible
    - If you can't find the chemical formula for a given acronym, return it as it is"""
    context: str = dspy.InputField()
    acronyms: list[str] = dspy.InputField()
    acronym_with_formula: dict[str, str] = dspy.OutputField(desc="dictionary of acronyms or labels and their full formula, e.g. {'LBO': 'LiB3O5}")

def cems_post_process(docs, cems, drop_rule, standard_rule):
    """
    Processes the extracted chemical names based on specified rules.

    Args:
        docs (list): A list of Document objects.
        cems (list): A list of extracted chemical names.
        drop_rule (str): A prompt specifying the rule for dropping invalid chemical names.
        standard_rule (str): A prompt specifying the rule for standardizing chemical names.

    Returns:
        tuple[Optional[dict], list]: A tuple containing:
            - An optional dictionary (if applicable).
            - A list of processed chemical names.
    """
    process_cems = dspy.ChainOfThought(ProcessChemicalNames)
    drops = []
    if drop_rule:
        drop_sig = DropInvalid
        drop_sig.instructions = drop_rule
        drop_invalid = dspy.ChainOfThought(drop_sig)
        drops = drop_invalid(cems=cems).dropped

    cems_dropped = [cem for cem in cems if cem not in drops]
    if not cems_dropped:
        return None, drops

    standard_dict = {cem: cem for cem in cems_dropped}
    if standard_rule:
        standard_sig = Standardization
        standard_sig.instructions = standard_rule
        standardize = dspy.ChainOfThought(standard_sig)
        standard_dict = standardize(cems=cems_dropped).standardized_cems
    standard_cems = list(standard_dict.values())
    
    acronyms = get_acronym_2_shots(cems=standard_cems).acronyms
    acronyms = [acronym for acronym in acronyms if acronym in standard_cems]
    if not acronyms:
        return standard_dict, drops

    # resolved acronyms
    patterns = [re.compile(f'\\b{acronym}\\b') for acronym in acronyms]
    def get_context(pattern):
        for doc in docs:
            if re.search(pattern, doc.page_content):
                return doc
        return None
    relevant_docs = list(filter(None, [get_context(pattern) for pattern in patterns]))
    title, abstracts = get_title_abs(docs)
    with_order = list(zip(docs, range(len(docs))))
    relevant_docs = reorder_docs(with_order, relevant_docs)
    context = render_docs(abstracts + relevant_docs, title, '')
    acronym_dict = process_cems(context=context, acronyms=acronyms).acronym_with_formula
    def get_resolved(standard_dict, acronym_dict):
        resolved = {}
        for cem, standard_cem in standard_dict.items():
            flag = False
            for acronym, formula in acronym_dict.items():
                if standard_cem == acronym: # when the standard cem is an acronym
                    resolved[cem] = formula
                    flag = True
                    break
            if not flag:
                resolved[cem] = standard_cem
        return resolved
    resolved = get_resolved(standard_dict, acronym_dict)
    return resolved, drops
    