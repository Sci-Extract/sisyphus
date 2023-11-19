"""
attention: the result jsonl file and standard json file considered to be different files. the standard one only contains one data.
main api: build_similarity, select_top_n. both return dataframe.
"""

import json
import os

import pandas as pd
import numpy as np
from pandas import DataFrame
from numpy import dot


# read the result of embedding and convert it to dataframe with columns: content, metadata, embedding.
def construct_df(jsonl_file_name) -> DataFrame:
    df = pd.DataFrame()
    bulk = []
    content = []
    metadata = []
    embedding = []
    with open(jsonl_file_name, encoding='utf-8') as file:
        for line in file:
            bulk.append(json.loads(line))
    for unit in bulk:
        # for unit, the basic structure is [request_json, response, metadata]
        content.append(unit[0]["input"])
        metadata.append(unit[2]["file_name"])
        embedding.append(np.array(unit[1]["data"][0]["embedding"]))
    df = pd.DataFrame(
        {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }
    )
    return df

def build_similarity(result_jsonl_file_name: str, standard_json_file_name: str, save_file_name: str|None) -> DataFrame:
    df = construct_df(result_jsonl_file_name)
    standard_vector = construct_df(standard_json_file_name)["embedding"][0]
    df["similarity"] = df["embedding"].apply(lambda vector: dot(vector, standard_vector))
    save_file_dir = "data"
    if save_file_name:
        save_file_path = os.path.join(save_file_dir, save_file_name + ".csv")
        df.to_csv(save_file_path, index=False)
    return df

def select_top_n(df: DataFrame, top_n: int) -> DataFrame:
    df_sorted = df.sort_values(by='similarity', ascending=False)
    grouped = df_sorted.groupby('metadata')

    def get_top_5(group):
        return group.head(top_n)

    top_5_values = grouped.apply(get_top_5)
    top_5_values = top_5_values.reset_index(drop=True)
    return top_5_values
