"""keyword for article search: organic solar cell* additivel
search engine: web of science
filter on: type=article, research areas=chemistry, material science, energy fuels, year=2015-now
total: 3541 results
take 90 from 1-1000 dois
"""
import re
from typing import Optional

import ujson
import dspy
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt

import sisyphus.patch.dspy_patch
from sisyphus.chain.chain_elements import Filter, Writer, run_chains_with_extarction_history_multi_threads
from sisyphus.chain.customized_elements import customized_extractor
from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb, return_valid

# config
lm = dspy.LM('openai/gpt-4o-mini', cache=False)
dspy.configure(lm=lm)
file_db = 'pvc_90'
result_db = 'pvc_90_result_less_code'

def read_examples(path):
    with open(path, 'r') as f:
        data = ujson.load(f)
    examples = data["examples"]
    return [dspy.Example(**e).with_inputs('context') for e in examples]

class SolarCell(BaseModel):
    solar_cell_name: str = Field(description='the name of the solar cell. e.g. PTB7:PC71BM')
    donor: str = Field(description='the donor material of the solar cell, usually on the left side of solar cell name')
    acceptor: str = Field(description='the acceptor material of the solar cell, usually on the right side of solar cell name')
    additive: Optional[str] = Field(description='the additive material of the solar cell, if any')
    pce: float = Field(description='power conversion efficiency quantified in percentage')

class PCE(dspy.Signature):
    """Extract solar cell compostions and correspond power conversion efficiency (PCE) from text"""
    context: str = dspy.InputField(desc='text might contain solar cell compositions and PCE')
    solar_cells: Optional[list[SolarCell]] = dspy.OutputField(desc='a list of solar cell compositions and PCE, can be null')

class Critical(dspy.Signature):
    """Does the prediction match the gold standard?"""
    gold: list[SolarCell] = dspy.InputField(desc='gold standard solar cell')
    pred: list[SolarCell] = dspy.InputField(desc='predicted solar cell')
    answer: bool = dspy.OutputField(desc='whether the prediction matches the gold standard, the sequence does not matter')

class Classifier(dspy.Signature):
    """Whether the text contains solar cell compositions and PCE value both"""
    context: str = dspy.InputField(desc='text might contain solar cell compositions and PCE')
    answer: bool = dspy.OutputField(desc='whether the text contains solar cell compositions and PCE value both') 

def critisize(gold, pred, trace=None):
    gold_sc = [SolarCell(**cell) for cell in gold.solar_cells]
    pred_sc = pred.solar_cells
    return dspy.ChainOfThought(signature=Critical)(gold=gold_sc, pred=pred_sc).answer
 
examples = read_examples('curated_pce_examples.json')

# load compiled extractor
compiled_extractor = dspy.ChainOfThought(PCE)
compiled_extractor.load('compiled_pce_extractor.json')

def my_filter(document):
    if match_pce_regex.search(document.metadata['sub_titles']):
        return True
    return False

# integrated with sisyphus pipeline
plaindb = get_plain_articledb(file_db)
filter_ = Filter(db=plaindb, with_abstract=True, filter_func=my_filter)
writer = Writer(get_create_resultdb(result_db, SolarCell))
classifier = dspy.ChainOfThought(Classifier)
match_pce_regex = re.compile(r'result|introduction', re.I)

@return_valid
# @retry(retry=retry_if_exception(ValidationError), stop=stop_after_attempt(2), retry_error_callback=lambda r: None)
def extract(doc):
    if classifier(context=doc).answer:
        text = doc.page_content
        abstract = doc.metadata['abstract']
        title = doc.metadata['title']
        sub_titles = doc.metadata['sub_titles']
        formatted_text = f'title: {title}\nabstract: {abstract}\nsection title: {sub_titles}\ntext: {text}'
        return compiled_extractor(context=formatted_text).solar_cells

extractor = customized_extractor(my_extractor=extract, mode='thread', concurrent_number=5)

chain = filter_ + extractor + writer
run_chains_with_extarction_history_multi_threads(chain, 'articles_processed', batch_size=10, namespace='pvc_90', extract_nums=3)
