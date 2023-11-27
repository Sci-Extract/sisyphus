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
        embedding.append(np.array(unit[1]["data"][0]["embedding"]))
    df = pd.DataFrame(
        {
            "content": content,
            "file_name": file_name,
            "task_id": task_id,
            "embedding": embedding
        }
    )
    return df

def build_similarity(result_jsonl_file_path: str, standard_vector: np.array, save_file_name: str|None) -> DataFrame:
    df = construct_df_embed(result_jsonl_file_path)
    df["similarity"] = df["embedding"].apply(lambda vector: dot(vector, standard_vector))
    save_file_dir = "data"
    if save_file_name:
        save_file_path = os.path.join(save_file_dir, save_file_name)
        df.to_csv(save_file_path, index=False)
    return df

def select_top_n(df: DataFrame, top_n: int, save_file_name: str|None) -> Tuple[DataFrame, str]:
    df_sorted = df.sort_values(by='similarity', ascending=False)
    grouped = df_sorted.groupby('file_name')

    def get_top_n(group):
        return group.head(top_n)

    top_5_values = grouped.apply(get_top_n)
    top_5_values = top_5_values.reset_index(drop=True)
    save_file_dir = "data"
    if save_file_name:
        save_file_path = os.path.join(save_file_dir, save_file_name)
        top_5_values.to_csv(save_file_path, index=False)
    return top_5_values, save_file_path

def construct_df_completion_cls(jsonl_file_path): # classify
    df = pd.DataFrame()
    bulk = []
    file_name = []
    task_id = []
    response = []
    with open(jsonl_file_path, encoding='utf-8') as file:
        for line in file:
            bulk.append(json.loads(line))
    for unit in bulk:
        # for unit, the basic structure is [request_json, response, metadata]
        file_name.append(unit[2]["file_name"])
        task_id.append(unit[2]["task_id"])
        raw_response = unit[1]["choices"][0]["message"]["content"]
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
    return df