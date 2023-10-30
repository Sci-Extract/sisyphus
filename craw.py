from sisyphus.crawler.elsevier_async import ElsevierCrawler
import httpx
import time

import asyncio
import pandas as pd

df = pd.read_csv("data/elsevier_doi.csv")
doi_ls = df["doi"].tolist()[:10]

els_api_key = "f6f2fb8ca79243871a31f1c3cd0b4204"
elseviercrawler = ElsevierCrawler(doi_ls, els_api_key, logging_level=20, max_requests_per_second=6)

async def main():
    start = time.time()
    await elseviercrawler.run()
    end = time.time()
    print(f"totally running time {end-start:.2f} s")

asyncio.run(main())
