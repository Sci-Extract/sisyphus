"""
attention: the result jsonl file and standard json file considered to be different files. the standard one only contains one data.
main api: build_similarity, select_top_n. both return dataframe.
"""

import json
import os
import re
from typing import Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame
from numpy import dot


# read the result of embedding and convert it to dataframe with columns: content, metadata, embedding.
def construct_df_embed(jsonl_file_path) -> DataFrame:
    df = pd.DataFrame()
    bulk = []
    content = []
    file_name = []
    task_id = []
    embedding = []
    with open(jsonl_file_path, encoding='utf-8') as file:
        for line in file:
            bulk.append(json.loads(line))
    for unit in bulk:
        # for unit, the basic structure is [request_json, response, metadata]
        content.append(unit[0]["input"])
        file_name.append(unit[2]["file_name"])
        task_id.append(unit[2]["task_id"])
        embedding.append(np.array(unit[1]["embedding"]))
    df = pd.DataFrame(
        {
            "content": content,
            "file_name": file_name,
            "task_id": task_id,
            "embedding": embedding
        }
    )
    return df

def build_similarity(result_jsonl_file_path: str, standard_vector: np.array, save_file: str|None) -> DataFrame:
    df = construct_df_embed(result_jsonl_file_path)
    df["similarity"] = df["embedding"].apply(lambda vector: dot(vector, standard_vector))
    if save_file:
        df.to_csv(save_file, index=False)
    return df

def select_top_n(df: DataFrame, top_n: int, save_file: str | None = None) -> DataFrame:
    df_sorted = df.sort_values(by='similarity', ascending=False)
    grouped = df_sorted.groupby('file_name')

    def get_top_n(group):
        return group.head(top_n)

    top_5_values = grouped.apply(get_top_n)
    top_5_values = top_5_values.reset_index(drop=True)
    if save_file:
        top_5_values.to_csv(save_file, index=False)
    return top_5_values

def get_candidates(jsonl_file_path, corpus_file: str):
    """Get candidates from classify results, corpus is the file has task_id and text content"""
    df = pd.DataFrame()
    bulk = []
    file_name = []
    task_id = []
    response = []
    with open(jsonl_file_path, encoding='utf-8') as file:
        for line in file:
            bulk.append(json.loads(line))
    for unit in bulk:
        # for each unit, the basic structure is [request_json, response, metadata]
        file_name.append(unit[2]["file_name"])
        task_id.append(unit[2]["task_id"])
        raw_response = unit[1]["content"]
        if isinstance(raw_response, str):
            if re.search(r'false', raw_response, re.I):
                response.append(False)
            else:
                response.append(True)
        elif isinstance(raw_response, dict):
            flag = False
            for key in raw_response:
                if raw_response[key] == 'False' or 'false' or False:
                    flag = True
                    break
            if flag:
                response.append(False)
            else:
                response.append(True)
    
    df = pd.DataFrame(
        {
            "file_name": file_name,
            "task_id": task_id,
            "response": response
        }
    )
    df = df[df['response'] == True]
    ids = df["task_id"].to_list()
    texts = []
    corpus_task_ids = []
    corpus = []
    with open(corpus_file, encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            corpus.append(d)
            corpus_task_ids.append(d["metadata"]["task_id"])
        for id in ids:
            index = corpus_task_ids.index(id)
            texts.append(corpus[index]["input"])
    df["content"] = texts
    
    return df