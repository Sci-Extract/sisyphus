import re
from typing import Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from pydantic import BaseModel

from sisyphus.chain.chain_elements import DocInfo
from sisyphus.utils.helper_functions import render_docs, get_title_abs, reorder_paras, render_docs_without_title

from .paragraph import Paragraph, ParagraphExtend
from .synthesis import get_processes_then_extract, ProcessingHistory
from .properties import property_extract_agents


class Result(BaseModel):
    property_name: str
    content: dict
    processes: Optional[list[dict]]

def extract_synthesis(paras: list[Paragraph]):
    syn_paras = [para for para in paras if para.is_synthesis]
    if not syn_paras:
        return
    last_intro_para = [doc for doc in paras if re.search(r'introduction', doc.metadata['sub_titles'], re.I)][-1:] # last para of intro often contains key information
    title, abs = get_title_abs(paras)
    with_context_paras = reorder_paras(abs + last_intro_para + syn_paras)
    
    syn_text = render_docs(with_context_paras, title)
    routes = get_processes_then_extract(syn_text)
    return routes

def get_relevant_paras(property_name, paras, include_context=True):
    rel_paras = [para for para in paras if para.has_property(property_name)]
    if not rel_paras:
        return
    if include_context:
        _, abs = get_title_abs(paras)
        last_intro_para = [doc for doc in paras if re.search(r'introduction', doc.metadata['sub_titles'], re.I)][-1:]
        return reorder_paras(abs + last_intro_para + rel_paras)
    return rel_paras

# def extract_properties(property_name: str, paras, history: ProcessingHistory):
#     extractor = property_extract_agents[property_name]
#     rel_paras = get_relevant_paras(property_name, paras)
#     if not rel_paras:
#         return
#     title, _ = get_title_abs(paras)
#     doi = paras[0].metadata['doi']
#     source = paras[0].metadata['source']
#     syn_paras = [para for para in paras if para.is_synthesis]
#     syn_text = render_docs_without_title(syn_paras)
#     r = extractor(syn_paragraph=syn_text, routes=history.serialize_history(), paragraph=render_docs(rel_paras, title)).properties
#     if not r:
#         return
#     def get_processes(c):
#         if c.meta:
#             return history.get_route_with_id(c.meta.route_id).get_processes_with_ids(c.meta.step_ids)
#         return None
#     def remove_meta(c):
#         c_dict = c.model_dump()
#         c_dict.pop('meta')
#         return c_dict
#     results = [Result(property_name=property_name, content=remove_meta(c), processes=get_processes(c)) for c in r]
#     extended_para = ParagraphExtend.merge_paras(rel_paras, metadata={'type': property_name, 'doi': doi, 'source': source}, title=title)
#     docinfo = DocInfo(extended_para, results)

#     return docinfo

def extract_properties_para_wise(property_name: str, paras: list[Paragraph], history: ProcessingHistory):
    rel_paras = [para for para in paras if para.has_property(property_name)]
    if not rel_paras:
        return

    title, _ = get_title_abs(paras)
    
    # get context
    syn_paras = [para for para in paras if para.is_synthesis]
    last_intro_para = [para for para in paras if re.search(r'introduction', para.metadata['sub_titles'], re.I)][-1:]
    composition_para = [para for para in paras if para.has_property('composition')]
    reordered = reorder_paras(syn_paras + last_intro_para + composition_para)
    context = render_docs(reordered, title, 'HEAs composition:')

    def get_processes(c):
        if c.meta:
            return history.get_route_with_id(c.meta.route_id).get_processes_with_ids(c.meta.step_ids)
        return None
    def remove_meta(c):
        c_dict = c.model_dump()
        c_dict.pop('meta')
        return c_dict

    # extract
    results = []
    extractor = property_extract_agents[property_name]
    with ThreadPoolExecutor(5) as executor:
        futures = [executor.submit(extractor, paragraph=para.page_content, context=context, routes=history.serialize_history()) for para in rel_paras]
        future_para = dict(zip(futures, rel_paras))
        for future in as_completed(futures):
            extraction = future.result()
            if r:=extraction.properties:
                info = [Result(property_name=property_name, content=remove_meta(c), processes=get_processes(c)) for c in r]
                results.append(DocInfo(future_para[future], info))

    return results

def extract_properties_bulk(property_name: str, paras: list[Paragraph], history: ProcessingHistory):
    rel_paras = [para for para in paras if para.has_property(property_name)]
    if not rel_paras:
        return

    title, _ = get_title_abs(paras)
    
    # get context
    syn_paras = [para for para in paras if para.is_synthesis]
    last_intro_para = [para for para in paras if re.search(r'introduction', para.metadata['sub_titles'], re.I)][-1:]
    composition_para = [para for para in paras if para.has_property('composition')]
    reordered = reorder_paras(syn_paras + last_intro_para + composition_para)
    context = render_docs(reordered, title, 'HEAs composition:')

    def get_processes(c):
        if c.meta:
            return history.get_route_with_id(c.meta.route_id).get_processes_with_ids(c.meta.step_ids)
        return None
    def remove_meta(c):
        c_dict = c.model_dump()
        c_dict.pop('meta')
        return c_dict

    # extract
    results = []
    extractor = property_extract_agents[property_name]
    paras_to_extract = reorder_paras([para for para in rel_paras if para not in reordered])
    text_to_extract = render_docs_without_title(paras_to_extract)
    extraction = extractor(paragraph=text_to_extract, context=context, routes=history.serialize_history())
    properties = extraction.properties
    if properties:
        para_extend = ParagraphExtend.merge_paras(paras_to_extract, metadata={'type': property_name, 'doi': paras[0].metadata['doi'], 'source': paras[0].metadata['source']}, title=title)
        info = [Result(property_name=property_name, content=remove_meta(c), processes=get_processes(c)) for c in properties]
        results.append(DocInfo(para_extend, info))

    return results

def extract_properties_without_history_para_wise(property_name: str, paras: list[Paragraph]):
    rel_paras = [para for para in paras if para.has_property(property_name)]
    if not rel_paras:
        return

    title, _ = get_title_abs(paras)
    
    # get context
    last_intro_para = [para for para in paras if re.search(r'introduction', para.metadata['sub_titles'], re.I)][-1:]
    composition_para = [para for para in paras if para.has_property('composition')]
    reordered = reorder_paras(last_intro_para + composition_para)
    if not reordered:
        context = ''
    else:
        context = render_docs(reordered, title, 'HEAs composition:')

    def remove_meta(c):
        c_dict = c.model_dump()
        c_dict.pop('meta')
        return c_dict

    # extract
    results = []
    extractor = property_extract_agents[property_name]
    with ThreadPoolExecutor(5) as executor:
        futures = [executor.submit(extractor, paragraph=para.page_content, context=context, routes='') for para in rel_paras]
        future_para = dict(zip(futures, rel_paras))
        for future in as_completed(futures):
            extraction = future.result()
            if r:=extraction.properties:
                info = [Result(property_name=property_name, content=remove_meta(c), processes=None) for c in r]
                results.append(DocInfo(future_para[future], info))

    return results

def extract_properties_without_history_bulk(property_name: str, paras: list[Paragraph]):
    rel_paras = [para for para in paras if para.has_property(property_name)]
    if not rel_paras:
        return

    title, _ = get_title_abs(paras)
    
    # get context
    last_intro_para = [para for para in paras if re.search(r'introduction', para.metadata['sub_titles'], re.I)][-1:]
    composition_para = [para for para in paras if para.has_property('composition')]
    reordered = reorder_paras(last_intro_para + composition_para)
    context = render_docs(reordered, title, 'HEAs composition:')

    def remove_meta(c):
        c_dict = c.model_dump()
        c_dict.pop('meta')
        return c_dict

    # extract
    results = []
    extractor = property_extract_agents[property_name]
    paras_to_extract = reorder_paras([para for para in rel_paras if para not in reordered])
    text_to_extract = render_docs_without_title(paras_to_extract)
    extraction = extractor(paragraph=text_to_extract, context=context, routes='')
    properties = extraction.properties
    if properties:
        para_extend = ParagraphExtend.merge_paras(paras_to_extract, metadata={'type': property_name, 'doi': paras[0].metadata['doi'], 'source': paras[0].metadata['source']}, title=title)
        info = [Result(property_name=property_name, content=remove_meta(c), processes=None) for c in properties]
        results.append(DocInfo(para_extend, info))


    return results


def extract(paras: list[Paragraph], extract_model: dspy.LM):
    """extract properties and associated processes for each record"""
    with dspy.context(lm=extract_model):
        routes = extract_synthesis(paras)
    if routes:
        history = ProcessingHistory(routes)
        extract_ = partial(extract_properties_without_history_bulk, paras=paras, history=history)
    else:
        extract_ = partial(extract_properties_bulk, paras=paras)

    data = []
    with ThreadPoolExecutor(5) as worker:
        futures = [worker.submit(extract_, property_name) for property_name in property_extract_agents]
        for future in futures:
            if result:=future.result():
                data.extend(result)
    return data
