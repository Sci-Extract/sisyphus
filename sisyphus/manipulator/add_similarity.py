import json

import pandas as pd
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
        embedding.append(unit[1]["data"][0]["embedding"])
    embedding = [ float(axis) for instance,axis in embedding]
    df = pd.DataFrame(
        {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }
    )
    return df

def df_builder(result_jsonl_file_name: str, standard_json_file_name: str, save_file_name: str|None) -> DataFrame:
    df = construct_df(result_jsonl_file_name)
    standard_vector = construct_df(standard_json_file_name)["embedding"]
    df["similarity"] = df["embedding"].apply(lambda vector: dot(vector, standard_vector))
    if save_file_name:
        df.to_csv(save_file_name + '.csv', index=False)
    return df
