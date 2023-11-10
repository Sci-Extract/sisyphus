import asyncio
import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from sisyphus.crawler.async_playwright import manager


_ = load_dotenv(find_dotenv())
els_api_key = os.getenv("els_api_key") 

df = pd.read_csv("data/doi_list.csv") # prepared doi file
doi_list: list[str] = df["doi"].dropna().unique().tolist()[:20]

asyncio.run(manager(doi_list, els_api_key=None))
