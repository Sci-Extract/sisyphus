"""
Async request to Elsevier API, please be cautious about the rate limitation which is 6 per second maximum.
Control the rate limit by Semaphore and for each task upon finished sleep for 1s.
Default retry attempts was 3.
"""

import asyncio
import logging
import os

import httpx


class ElsevierCrawler:
    """
    Fetch fetch article from elsevier asynchronously.

    :param logging_level: 10 for debug level, 20 for info, and so on
    """
    def __init__(
        self, 
        doi_list: list, 
        els_api_key: str, 
        logging_level: int, 
        max_requests_per_second=6
    ):
        self.doi_list = doi_list
        self.els_api_key = els_api_key
        self.max_requests_per_second = max_requests_per_second
        logging.basicConfig(level=logging_level)

    async def run(self, save_file_dir="data/elsevier"):
        # if not then create the dir
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)

        async with httpx.AsyncClient(verify=False) as client:
            sema = asyncio.Semaphore(self.max_requests_per_second)
            tasks = [asyncio.create_task(self.fetch(client, doi, sema, self.create_path(save_file_dir, doi.split("/")[1], suffix=".xml")))
            for doi in self.doi_list]
            await asyncio.gather(*tasks)

    async def fetch(self, client, doi, sema:asyncio.Semaphore, store_file_loc):
        url='https://api.elsevier.com/content/article/doi/' + doi + '?view=FULL'
        headers = {
          'X-ELS-APIKEY': self.els_api_key,
          'Accept': 'text/xml'
        }
        retry_counter = 3

        async with sema:
            while retry_counter:
                try:
                    response = await client.get(url, headers=headers)

                    if response.status_code != 200:
                        retry_counter -= 1
                        await asyncio.sleep(15)
                        logging.info(f"failed with {doi}, retry left: {retry_counter}")
                        break

                    with open(store_file_loc, 'w', encoding='utf-8') as file:
                        logging.info(f"successfully download {doi}")
                        file.write(response.text)

                    await asyncio.sleep(1)
                    break

                except Exception as e:
                    raise e

    def create_path(self, *path, suffix: str):
        path = os.path.join(*path) + suffix
        return path
