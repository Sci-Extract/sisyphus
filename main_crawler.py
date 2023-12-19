import asyncio
import argparse
import os
import re

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from sisyphus.crawler.async_playwright import manager


_ = load_dotenv(find_dotenv())
els_api_key = os.getenv("els_api_key") 

# df = pd.read_csv("data_forcrawler_test\\doi_list.csv") # prepared doi file
# doi_list: list[str] = df["doi"].dropna().unique().tolist()[:5]

# asyncio.run(manager(doi_list, els_api_key=els_api_key))

def parse_file(file) -> list[str]:
    doi_ls = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            if line.startswith("http"):
                doi_ls.append(line.strip('\n').replace("https://doi.org/", ''))
            if re.search(r'^\d{2}\.\d{4}/.+', line):
                doi_ls.append(line.strip('\n'))
    return doi_ls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_dois", help=".txt file contains dois you want to retrieve")
    args = parser.parse_args()

    doi_ls = parse_file(args.retrieval_dois)
    doi_ls = list(set(doi_ls))
    asyncio.run(manager(doi_ls, els_api_key=els_api_key))
