import re
import os
from typing import Literal, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb
from sisyphus.chain.chain_elements import Filter, Writer, DocInfo, run_chains_with_extarction_history_multi_threads


PAPER_DB = 'heas_764'
K = 3
QUERY_SYN = """Experimental procedures describing the synthesis and processing of HEAs materials, including methods such as melting, casting, rolling, annealing, heat treatment, or other fabrication techniques. Details often include specific temperatures (e.g., °C), durations (e.g., hours, minutes), atmospheric conditions (e.g., argon, vacuum), mechanical deformation (e.g., rolling reduction)."""
QUERY_MECHANICAL = "The stress-strain curve of alloy, describes yield strength (ys), tensile strength (uts) and elogation properties, for example, CoCuFeMnNi shows tensile strength of 1300 MPa and total elongation of 20%"
QUERY_PHASE = """Microstructure characterization of alloys (common phases include FCC, BCC, HCP, L12, B2 etc.), usually througth technique like XRD or TEM. Describe about phase and grain size and bouondaries"""
SYSTEM_MESSAGE = """You are an expert at structured data extraction from HEAs (high entropy alloys) domain. You will be given unstructured text from a research paper and should convert it into the given structure"""
INSTRUCTION =  """Extract all HEAs tensile and compressive properties and their synthesis methods from text, including HEAs refered in the paper.
Note:
For HEAs composition
- HEAs composition should be strictly comply with standard naming convention in at.%, e.g Hf0.5Mo0.5NbTiZrC0.3. If the alloy is doped or added with other element and the author did not provide a nominal composition, you should give name such as W-CoCrFeMnNi. Acronym such as HEA-1 is prohibited.
For mechanical value
- If the value are provided as average +/- standard deviation, use the average value, e.g 700 +/- 30 should be recorded as 700. If the value is provided as a range, use the lower value, e.g 500-600 should be recorded as 500.
- Prioritize the value in the table over text if there is a conflict.
For synthesis
- As-cast (remelting serval times and casting) materials should not contains thermal mechanical processes.
- Correctly associate synthesis parameters with the corresponding mechanical properties.
"""


class AlloyRecord(BaseModel):
    composition: str = Field(description='The nominal chemical composition of the alloy in at.%. Ensure the validity of the formula, e.g. Hf0.5Mo0.5NbTiZrC0.3.')
    phase: Optional[str] = Field(description='The phase of the alloy, such as FCC, BCC, HCP, L12, B2, Laves etc. If there are multiple phases, separate them with commas')
    ys: Optional[float] = Field(description='the value of yield strength, convert to MPa if the unit is not MPa, e.g. 1 GPa -> 1000 MPa')
    uts: Optional[float] = Field(description='the value of ultimate tensile strength (for tensile tests) or maximum compression strength (for compression tests), convert to MPa if the unit is not MPa, e.g. 1GPa -> 1000 MPa')
    strain: Optional[float] = Field(description='the value of plastic elongation or plastic compression strain, convert to percentage if the unit is not percentage, e.g. 1%')
    fabrication: Optional[str] = Field(description='The fabrication method of the alloy, e.g. vacuum arc-melting')
    thermal_mechanical_processings: Optional[str]  = Field(description='The sequential post-processing steps of the alloy separated by vertical bar "|", be briefly, eg., annealed at 900 °C for 4 h | homogenized at 1200 °C for 2 h')

    test_type: Literal['tensile', 'compressive']

    test_temperature: str = Field(description='The temperature at which the mechanical properties were tested, e.g. 25 °C. If the temperature is given in Kelvin, convert it to Celsius by subtracting 273. If not mentioned in the text, record it as 25 °C.')

class Records(BaseModel):
    records: Optional[List[AlloyRecord]] = Field(description='The records of the alloy properties')


lm = dspy.LM('openai/gpt-4o-mini', cache=False, temperature=0)
dspy.configure(lm=lm)
model = ChatOpenAI(model='gpt-4o', temperature=0.0)
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = Chroma(collection_name='hea_collection', embedding_function=embeddings)
article_db = get_plain_articledb(PAPER_DB)
extract_template = ChatPromptTemplate(
    [
        ('system', SYSTEM_MESSAGE),
        ('user', '[START OF PAPER]{paper}[END OF PAPER]\n\nInstruction:\n{instruction}')
    ]
)
distill_extract_chain = extract_template | model.with_structured_output(Records, method='json_schema')

def retrieve(vector_store, source, query, sub_titles, k):
    if not sub_titles:
        filter_ = {"source": source}
    else:
        filter_ = {"$and":[{
            "sub_titles": {
                "$in": sub_titles
            }},
            {"source": source
        }]}
    return vector_store.similarity_search(
        query,
        k=k,
        filter=filter_
    )

def match_subtitles(docs, pattern):
    sub_titles = list(set([doc.metadata["sub_titles"] for doc in docs]))
    target_titles = []
    for title in sub_titles:
        if pattern.search(title):
            target_titles.append(title)
    return target_titles

def get_target_para_lm(source, sub_titles, query, classifier, class_, k=K, vector_store=vector_store):
    candidates = retrieve(vector_store, source, query, sub_titles, k)
    final = []
    with ThreadPoolExecutor(5) as worker:
        futures = [worker.submit(classifier, paragraph=candidate.page_content) for candidate in candidates]
        future_doc = dict(zip(futures, candidates))
        for future in as_completed(futures):
            if future.result().topic == class_:
                final.append(future_doc[future])
    return final

def get_target_para_regex(source, sub_titles, query, regex_pattern, k=K, vector_store=vector_store):
    candidates = retrieve(vector_store, source, query, sub_titles, k)
    final = [c for c in candidates if re.search(regex_pattern, c.page_content)]
    return final

def get_tables(docs):
    tables = [doc for doc in docs if doc.metadata["sub_titles"] == "table"]
    final = []
    with ThreadPoolExecutor(5) as worker:
        futures = [worker.submit(classifier_table, paragraph=table.page_content) for table in tables]
        future_doc = dict(zip(futures, tables))
        for future in as_completed(futures):
            pred = future.result()
            if pred.contain and pred.label == 'experimental':
                final.append(future_doc[future])
    return final

class ClassifySyn(dspy.Signature):
    """assign topic to paragraphs of HEAs(high entropy alloys) papers. The topics include synthesis, characterization, and others.
    Note: a qualified synthesis paragraph should include the synthesis and processing of materials, including methods such as melting, casting, rolling, annealing, heat treatment.be very strict about your decision."""
    paragraph: str = dspy.InputField()
    topic: Literal['synthesis', 'characterization', 'others'] = dspy.OutputField()

classifier_syn = dspy.ChainOfThought(signature=ClassifySyn)

class ClassifyTable(dspy.Signature):
    """assign labels to tables inside HEAs(high entropy alloys) papers. First classify the table to experimental or theoretical, then inspecting the table to find whether it contains properties relevant to tensile or compressive."""
    paragraph: str = dspy.InputField()
    label: Literal['experimental', 'theoretical'] = dspy.OutputField(desc="whether the contents of table are experimental measured or theoretical calculated.")
    contain: bool = dspy.OutputField(desc='whether the table includes any of these values: yield strength, tensile/compressive strength, or strain of HEAs')

classifier_table = dspy.ChainOfThought(signature=ClassifyTable)

class ClassifyPaper(dspy.Signature):
    """assign label to HEAs (high entropy alloys) paper based on their title and abstract."""
    title: str = dspy.InputField()
    abstract: str = dspy.InputField()
    label: Literal['hea_experimental', 'hea_theoretical', 'irrelevant'] = dspy.OutputField(desc="Pay attention to keywords such as 'molecular dynamics' or 'machine learning,' which should be labeled as hea_theoretical. Label keywords related to fabrication processes as hea_experimental.")
    mechanical_relevancy: bool = dspy.OutputField(desc='whether this paper describe the mechanical properties such as tensile or compressive')

classifier_paper = dspy.ChainOfThought(signature=ClassifyPaper)

def get_meta_info(docs):
    metadata = docs[0].metadata
    source = metadata['source']
    title = metadata['title']
    doi = metadata['doi']
    return source, title, doi

def reorder_docs(ordered, docs):
    """reorder the retrieved documents.
    Note: this function can deal with duplication in the docs!
    WARNING: do not modify the docs"""
    with_order = []
    used_i = []
    for doc in docs:
        for o_doc, i in ordered:
            if i in used_i:
                continue
            if doc == o_doc:
                with_order.append((doc, i))
                used_i.append(i)
                break
    final = sorted(with_order, key=lambda x: x[1])
    return [el[0] for el in final]

def render_docs(docs, title):
    """render docs to nicely formatted paper look.
    Since the tables are the most information dense format, we put it at the tail"""
    tables = [doc for doc in docs if doc.metadata['sub_titles'] == 'table']
    paras = [doc for doc in docs if doc.metadata['sub_titles'] != 'table']

    previous_titles = []
    scratch_pad = [title]
    for para in paras:
        sub_titles = para.metadata['sub_titles'].split('/')
        title_to_write = [title for title in sub_titles if title not in previous_titles]
        previous_titles = sub_titles
        rendered_text = '\n'.join(title_to_write + [para.page_content])
        if title_to_write:
            rendered_text = '\n' + rendered_text
        scratch_pad.append(rendered_text)

    if tables:
        scratch_pad.append('\nMechanical property tables')
    for table in tables:
        scratch_pad.append('\n' + table.page_content)

    return '\n'.join(scratch_pad)

def distill_extract(docs):
    with_order = list(zip(docs, range(len(docs))))
    source, title, doi = get_meta_info(docs)
    abstract_docs = [doc for doc in docs if doc.metadata['sub_titles'] == 'Abstract']
    last_intro_doc = [doc for doc in docs if re.search(r'introduction', doc.metadata['sub_titles'], re.I)][-1:] # last para of intro often contains key information

    syn_pattern = re.compile(r'(experiment)|(preparation)|(method)', re.I)
    res_pattern = re.compile(r'result', re.I)
    mech_pattern = re.compile(r'(\b(MPa|GPa)\b|\d+(\.\d)?\s*%)')
    phase_pattern = re.compile(r'\b(FCC|BCC|HCP|L12|B2|Laves|f.c.c.|b.c.c.|h.c.p.|face-centered cubic|body-centered cubic|hexagonal close-packed)\b')
    sub_titles_syn = match_subtitles(docs, syn_pattern)
    sub_titles_res = match_subtitles(docs, res_pattern)

    # classify paper
    pred = classifier_paper(title=title, abstract='\n'.join(doc.page_content for doc in abstract_docs))
    if not (pred.mechanical_relevancy and pred.label == 'hea_experimental'):
        return

    # when there is only abstract available
    if len(docs) == 1:
        if mech_pattern.search(docs[0].page_content):
            return (docs[0], distill_extract_chain.invoke({'paper': render_docs(docs, title=title), 'instruction': INSTRUCTION}).records)
        return
    
    # embedding docs
    vector_store.add_documents(docs)
    
    # retain relevant paragraphs then extraction
    tensile_docs = get_target_para_regex(source, sub_titles_res, QUERY_MECHANICAL, mech_pattern, k=5)
    tensile_tables = get_tables(docs)
    if not (tensile_docs or tensile_tables):
        return # no tensile relevant text/tables found
    synthesis_docs = get_target_para_lm(source, sub_titles_syn, QUERY_SYN, classifier_syn, 'synthesis')
    phase_docs = get_target_para_regex(source, sub_titles_res, QUERY_PHASE, phase_pattern)
    disordered_docs = tensile_docs + tensile_tables + synthesis_docs + phase_docs + abstract_docs + last_intro_doc
    ordered_docs = reorder_docs(with_order, disordered_docs)
    paper = render_docs(ordered_docs, title)

    results = distill_extract_chain.invoke({'paper': paper, 'instruction': INSTRUCTION})
    doc = Document(page_content=paper, metadata={'source': source, 'doi': doi})
    return doc, results.records

def return_valid(t):
    if t is None:
        return
    doc, result = t
    if result:
        return [DocInfo(doc, result)]
    return

docs_getter = Filter(article_db)
# result_db = get_create_resultdb('heas_764_dist', AlloyRecord)
result_db = get_create_resultdb('my_test', Records)
chain = docs_getter + distill_extract + return_valid + Writer(result_db)
# inspect_chain = docs_getter + distill_extract

# def inspect_articles(articles):
#     with ThreadPoolExecutor(10) as worker:
#         futures = [worker.submit(inspect_chain.compose, article) for article in articles]
#         future_doc = dict(zip(futures, articles))
#         for future in tqdm(as_completed(futures), total=len(articles)):
#             if paper:=future.result():
#                 file_name = future_doc[future].strip('.html') + '.txt'
#                 file_path = os.path.join('inspect', file_name)
#                 with open(file_path, 'w', encoding='utf8') as f:
#                     f.write(paper)
    
# from random import sample
# articles_20 = sample(os.listdir('articles_processed'), 20)
# inspect_articles(articles_20)

# articles_20 = os.listdir('articles_processed_300')[:20]
# run_chains_with_extarction_history_multi_threads(chain, 'articles_processed', 5, 'heas_764_dist')
chain.compose('10.1016&sol;j.vacuum.2024.113026.html')
