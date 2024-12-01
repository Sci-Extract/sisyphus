"""keyword for article search: organic solar cell* additivel
search engine: web of science
filter on: type=article, research areas=chemistry, material science, energy fuels, year=2015-now
total: 3541 results
take 90 from 1-1000 dois
"""
import re
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import ujson
import dspy
from pydantic import BaseModel, Field

from sisyphus.chain.chain_elements import Filter, Writer, DocInfo
from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb

# config
lm = dspy.LM('openai/gpt-4o-mini', cache=False)
dspy.configure(lm=lm)
file_db = 'pvc_90'
result_db = 'pvc_90_result'

# patch
def custom_save(self, path, save_field_meta=False):
    def convert_to_dict(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(i) for i in obj]
        elif isinstance(obj, dspy.Example):
            d = dict(obj)
            return convert_to_dict(d)
        else:
            return obj

    state = self.dump_state(save_field_meta)
    state = convert_to_dict(state)
    with open(path, "w") as f:
        f.write(ujson.dumps(state, indent=2))

dspy.Module.save = custom_save

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
# zero_shot_extractor = dspy.ChainOfThought(signature=PCE)
# prediction = zero_shot_extractor(context=examples[0]['context']).solar_cells
# print(prediction)

# compile first
# optimizer = dspy.BootstrapFewShot(metric=critisize, max_bootstrapped_demos=1)
# compiled_extractor = optimizer.compile(dspy.ChainOfThought(signature=PCE), trainset=examples)
# compiled_extractor.save('compiled_pce_extractor.json')

# load compiled extractor
compiled_extractor = dspy.ChainOfThought(PCE)
compiled_extractor.load('compiled_pce_extractor.json')

# integrated with sisyphus pipeline
plaindb = get_plain_articledb(file_db)
filter_ = Filter(db=plaindb, with_abstract=True)
writer = Writer(get_create_resultdb(result_db, SolarCell))
classifier = dspy.ChainOfThought(Classifier)
match_pce_regex = re.compile(r'result|introduction', re.I)

def return_valid(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result if result else None
    return wrapper

@return_valid
def extract(doc):
    if classifier(context=doc).answer:
        text = doc.page_content
        abstract = doc.metadata['abstract']
        title = doc.metadata['title']
        sub_titles = doc.metadata['sub_titles']
        formatted_text = f'title: {title}\nabstract: {abstract}\nsection title: {sub_titles}\ntext: {text}'
        return compiled_extractor(context=formatted_text).solar_cells

@return_valid
def customized_filter(documents):
    docs = []
    for doc in documents:
        if match_pce_regex.search(doc.metadata['sub_titles']):
            docs.append(doc)
    return docs

def customized_extractor(docs):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(extract, docs)
    zipped_results = filter(lambda x: x[1], zip(docs, results))
    doc_infos = [DocInfo(doc=doc, info=result) for doc, result in zipped_results]
    return doc_infos

chain = filter_ + customized_filter + customized_extractor + writer
files = os.listdir('articles_processed')

from tqdm import tqdm
import time
start = time.time()
files.pop(0)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(chain.compose, file) for file in files]
    for future in tqdm(as_completed(futures), total=len(files)):
        future.result()
end = time.time()
print(f'time elapsed: {end - start:.2f}s')
