import asyncio
import argparse
import os
import re

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from sisyphus.crawler.async_playwright import manager
from sisyphus.utils.utilities import read_wos_excel


_ = load_dotenv(find_dotenv())
els_api_key = os.getenv("els_api_key") 

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
    parser.add_argument("doi_file", help=".txt/.xlsx file contains dois you want to retrieve")
    parser.add_argument('-r', '--rate', default=0.15, help='the rate of single crawler (rsc, wiley, acs...), default to 0.15, increase the number if you want to speed up, not recommand!')
    args = parser.parse_args()

    file: str = args.doi_file
    file_extension = os.path.splitext(file)[1]
    if file_extension == '.txt':
        doi_ls = parse_file(file)
    elif file_extension == '.xls':
        doi_ls = read_wos_excel(file)
    else:
        print("file format not supported")

    doi_ls = list(set(doi_ls))
    
    # prohibit aaas artilces due to the restriction rule
    doi_ls = [doi for doi in doi_ls if not doi.startswith('10.1126')]
    # doi_ls = [doi for doi in doi_ls if doi.startswith('10.1016')]
    asyncio.run(manager(doi_ls, els_api_key=els_api_key, rate_limit=float(args.rate)))
    print('We decide to ban aaas for now, maybe fix later')
