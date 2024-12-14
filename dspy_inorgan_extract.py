# maybe I can combine NER, RE with balance equation and validate again in one program
# I decide to abort this thought which will invoke too many separeted calls. It was unwise and hard to debug of course.
import re
import os
import json
from typing import Optional, Literal
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor

import dspy
from pydantic import BaseModel, Field

from sisyphus.chain import Filter, Writer
from sisyphus.chain.chain_elements import DocInfo
from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb


additional_args = ContextVar('additional_args', default={})
# config
lm = dspy.LM('openai/gpt-4o', cache=False)
dspy.configure(lm=lm)
# ARTICLE = 'inorganic_dspy'
ARTICLE = '40_with_good_title'
# TARGET = 'dspy_inorganic_4o_mini'
TARGET = 'test_single_replace_r_re'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    train_set = [dspy.Example(**example).with_inputs('text') for example in data['examples'][:10]]
    dev_set = [dspy.Example(**example).with_inputs('text') for example in data['examples'][10:]]
    return train_set, dev_set


class ClassifyReaction(dspy.Signature):
    """Giving the availability of extracting solid-state reaction formula from the text """
    text = dspy.InputField(desc='a piece of text which may contains solid-state chemical reaction formula')
    solid_state_definition = dspy.InputField(desc='the definition of solid-state reaction')
    
    extraction_availability_solid_react: int = dspy.OutputField(desc='1 for yes, 0 for no')


class Classify_CoT(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(signature=ClassifyReaction)

    def forward(self, text, solid_state_definition):
        prediction = self.predictor(text=text, solid_state_definition=solid_state_definition)
        return prediction
    

class Target(BaseModel):
    target_formula: str = Field(description='the formula of the target product, make sure it is a valid chemical formula')
    amount_var: dict[str, list[float]] = Field(description='the amount variable in the formula, e.g. AxBC, {x: [1, 2]}')
    extra_description: Optional[str] = Field(description='extra description other than the formula')

class Reaction(BaseModel):
    precursors: list[str] = Field(description='the precursors or starting material of reaction, ensure it is a valid chemical formula')
    additives: list[str] = Field(description='the additives of the reaction')
    target: Target = Field(description='the product of the reaction')
    reaction_type: Literal['solid-state', 'sol-gel', 'co-precipitation', 'hydrothermal', 'flux', 'others'] = Field(description='the type of the reaction')


class QA(dspy.Signature):
    """extract all chmemical reactions consitituent from the text, for reactions with element variables, please subsitute them with the corresponding element names"""
    text: str = dspy.InputField(desc='a piece of text which may contains chemical reactions')
    reactions: Optional[list[Reaction]] = dspy.OutputField(desc='the reactions extracted from the text, return null if no reaction found')

class ExtractReactionWithType(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(signature=QA)

    def forward(self, text):
        prediction = self.predictor(text=text)
        return prediction
        
    
class Assess(dspy.Signature):
    """Assess the quality of the extracted reaction"""
    grounded_result = dspy.InputField(desc='the grounded result by a human expert in json format')
    extracted_result = dspy.InputField(desc='the extracted result by a NLP program in json format')
    question = dspy.InputField()
    answer = dspy.OutputField(desc='please answer yes/no')


def llm_metric(gold, pred, trace=None):
    grounded_result_restrict = [{'precursors': r['precursors'], 'target': r['target']} for r in gold.reactions]
    extract_result_restrict = [{'precursors': r.precursors, 'target': r.target} for r in pred.reactions]
    critic = dspy.ChainOfThoughtWithHint(Assess)(
        grounded_result=json.dumps(grounded_result_restrict, indent=2),
        extracted_result=json.dumps(extract_result_restrict, indent=2),
        question='based on the given grounded result, do you think that the extracted result successfully capture the reaction?'
    )
    return True if critic.answer.lower() == 'yes' else False


class ResolveAbbreviations(dspy.Signature):
    """Resolves abbreviations for chemical terms and identifiers relevant to chemical reaction(s) to their chemical formulas with given context."""
    context: str = dspy.InputField(desc='the context where the chemical terms or identifiers appear')
    reactions: str = dspy.InputField(desc='the chemical reaction in which the chemical terms or identifiers appear')
    abbreviations: list[str] = dspy.InputField(desc='the abbreviations or inappropriate chemical terms of reactions')

    resolved_abbrevs: Optional[dict[str, str]] = dspy.OutputField(desc='the resolved abbreviations with their chemical formulas, e.g. {"1a": "1-phenyl-2-propanone"}')

class AbbreviationResolver(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(signature=ResolveAbbreviations)
    def forward(self, context, reactions, abbreviations):
        prediction = self.predictor(context=context, reactions=reactions, abbreviations=abbreviations)
        return prediction

class GetAbbreviations(dspy.Signature):
    
    chemical_terms: list[str] = dspy.InputField()

    non_valid_terms: Optional[list[str]] = dspy.OutputField(desc='the non-valid terms in the chemical_terms')

fs_abbrev_getter = dspy.LabeledFewShot(k=2).compile(
    dspy.ChainOfThought(signature=GetAbbreviations),
    trainset=[
        dspy.Example(chemical_terms=['Th(NO3)4·5H2O', 'MoO3', 'unknown product'], non_valid_terms=['unknown product']).with_inputs('chemical_terms'),
        dspy.Example(chemical_terms=['Ca(OH)2', 'Mg(OH)2', 'H3PO4', 'WH'], non_valid_terms=['WH']).with_inputs('chemical_terms'),
        ]
)

exp_section_pattern = re.compile(r'\b(?:experiment(?:al|s|ing|ed)?|synthesis(?:es|ing|ed)?|preparation(?:s|al|ed|ing)?|process(?:es|ion|ing)?|method(?:s)?)\b', re.I)
def filter_with_kw(doc):
    return bool(exp_section_pattern.search(doc.metadata['sub_titles']))

article_db = get_plain_articledb(ARTICLE)
article_getter = Filter(article_db, filter_func=filter_with_kw, with_abstract=True)
result_db = get_create_resultdb(TARGET, Reaction)

from sisyphus.utils.helper_functions import return_valid


cot = ExtractReactionWithType()
cot.load('compiled_direct')
compiled_extractor = cot
compiled_extractor.predictor.signature = QA

with open('resolve_examples.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    examples = [dspy.Example(**example).with_inputs('context', 'reactions') for example in data['examples'] if example['gpt_gen']]

resolver = AbbreviationResolver()
resolver.load('compiled_resolver_bootstrapped.json')
compiled_resolver = resolver

classifier = Classify_CoT()
definition = 'Solid-state reaction refers to a conventional method used in chemistry to synthesize various materials like ceramics and crystals by heating a mixture of raw materials in solid form.'


@return_valid
def extract(doc):
    if classifier(text=doc.page_content, solid_state_definition=definition).extraction_availability_solid_react:
        text = doc.page_content
        title = doc.metadata['title']
        abstract = doc.metadata['abstract']
        sub_titles = doc.metadata['sub_titles']
        context = f'title:\n{title}\n' + f'abstract:\n{abstract}\n' + 'subtitles:\n' + '\n'.join([f'{sub_title}' for sub_title in sub_titles.split('/')]) + f'\nparagraph:\n{text}'
        prediction = compiled_extractor(text=context)
        reactions = prediction.reactions
        return reactions

from sisyphus.chain.customized_elements import customized_extractor

my_extractor = customized_extractor(extract, 'thread', 5)

@return_valid
def get_abbrevs(docinfos: list[DocInfo]):
    abbrevs_all = set()
    
    def process_docinfo(docinfo):
        results = docinfo.info
        chem_terms_ls = [r.precursors + [r.target] for r in results]
        for chem_terms in chem_terms_ls:
            abbrevs = fs_abbrev_getter(chemical_terms=chem_terms).non_valid_terms
            if abbrevs:
                abbrevs_all.update(abbrevs)
        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     predictions = executor.map(fs_abbrev_getter, chem_terms_ls)
        # for prediction in predictions:
        #     abbrevs = prediction.non_valid_terms
        #     if abbrevs:
        #         abbrevs_all.update(abbrevs)
        
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_docinfo, docinfos)
    print('abbreviations:', abbrevs_all)
    args = additional_args.get()
    args.update({'abbrevs': abbrevs_all})
    additional_args.set(args)
    return docinfos


def format_reactions_repr(docinfo):
    _, results = docinfo.doc, docinfo.info
    reactions = [' + '.join(r.precursors) + ' -> ' + r.target for r in results]
    return '\n'.join(reactions)

def format_context(docinfos):
    paras = '\n'.join([doc.metadata['sub_titles'] + '\n' +  doc.page_content for doc, _ in docinfos])
    title = docinfos[0].doc.metadata['title']
    abstract = docinfos[0].doc.metadata['abstract']
    context = f'title: {title}'\
                f'\nabstract: {abstract}'\
                f'\nparagraphs: {paras}'
    reactions_repr = '\n'.join([format_reactions_repr(docinfo) for docinfo in docinfos])

    return context, reactions_repr

def replace_with_full_name(docinfo, resolved_abbrevs):
    doc, results = docinfo.doc, docinfo.info
    replaced_results = []
    for r in results:
        precursors = [resolved_abbrevs.get(p, p) for p in r.precursors]
        target = resolved_abbrevs.get(r.target, r.target)
        replaced_results.append(Reaction(precursors=precursors, target=target, reaction_type=r.reaction_type, additives=r.additives))
    return DocInfo(doc=doc, info=replaced_results)

@return_valid
def customized_validator(docinfos):
    if not additional_args.get()['abbrevs']:
        return docinfos # no abbrev, so no need to resolve
    context, reaction_repr = format_context(docinfos)
    prediction = compiled_resolver(context=context, reactions=reaction_repr, abbreviations=list(additional_args.get()['abbrevs']))
    resolved_abbrevs = prediction.resolved_abbrevs
    print('resolved abbreviations:', resolved_abbrevs)
    if not resolved_abbrevs: # no abbrev found
        return docinfos
    replaced_docinfos = [replace_with_full_name(docinfo, resolved_abbrevs) for docinfo in docinfos]
    return replaced_docinfos

# chain = article_getter + customized_filter + customized_extractor + get_abbrevs + customized_validator + Writer(result_db=result_db)
chain = article_getter + my_extractor + Writer(result_db=result_db)
# chain_with_out_writer = article_getter + my_extractor

import time
start = time.time()
from sisyphus.chain.chain_elements import run_chains_with_extarction_history_multi_threads
run_chains_with_extarction_history_multi_threads(chain, 'articles_processed', 10, 'dspy_6_shot_inorgan_recipes')
# result_db.clear_tables()
# with dspy.context(lm=dspy.LM('openai/gpt-4o-mini')):
docinfos = chain.compose('10.1021&sol;cm980126j.html')
# end = time.time()
# print('time:', end - start)
extracted = result_db.load_as_json(with_doi=True)
with open('replace_R.json', 'w', encoding='utf8') as f:
    json.dump(extracted, f, indent=2, ensure_ascii=False)

print(lm.inspect_history(2))
# for d in ds:
    # print(d)