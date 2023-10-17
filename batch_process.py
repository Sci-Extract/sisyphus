"""Request to crossref API for articles title information, the rate limit is 50 visits /s"""

import asyncio
import os
import time

import httpx
import pandas as pd

import sisyphus.utils.log


logger = sisyphus.utils.log.log("log.txt")
PREFIX = "https://api.crossref.org/works/"
METRICS = 1

class CrossRef:
    def __init__(self, dataframe, batch_size, out_file):
        self.dataframe = dataframe
        self.dois = dataframe.DOI
        self.batch_size = batch_size
        self.out_file = out_file
        self.url_title_json = list()
        
    def doi2url(self) -> list:
        urls = [PREFIX + doi for doi in self.dois.tolist()]
        return urls
    
    async def run(self):
        urls = self.doi2url()
        tasks = list()
        start_time = time.time()

        for url in urls:
            tasks.append(self.fetch(url))
            if len(tasks) == self.batch_size:
                # run batch
                await asyncio.gather(*tasks)
                self.save()
                tasks.clear()

                elapsed_time = time.time() - start_time
                if elapsed_time < METRICS:
                    await asyncio.sleep(METRICS - elapsed_time)
                
            start_time = time.time()
            
        await asyncio.gather(*tasks)
        self.save()

    async def fetch(self, url, retry_attempts=3):
        remain_attempts = retry_attempts
        while remain_attempts:
            async with httpx.AsyncClient(verify=False) as client:
                try:
                    response = await client.get(url)
                    title = self.collect_title(response)
                    self.sync(url, title)
                    logger.info(f"successfully reach to {url}")
                    break

                except httpx.RequestError as e:
                    logger.error(f"request to {url} failed, retrying")
                    remain_attempts -= 1
                    await asyncio.sleep(1)

                except KeyError:
                    logger.error(f"nothing found in {url}")
                    break

        if not remain_attempts:
            logger.critical(f"{retry_attempts} times retrying failed, please retry later")

    def collect_title(self, response):
        title = response.json()["message"]["title"][0]
        return title
    
    def sync(self, url, title):
        self.url_title_json.append({"url": url, "title": title})
    
    def save(self):
        url_title_df = pd.DataFrame(self.url_title_json)
        if not os.path.exists(self.out_file):
            url_title_df.to_csv(self.out_file, index=False, header=True)
        else:
            url_title_df.to_csv(self.out_file, index=False, mode='a', header=False)
        # clear, indicate that duplicate save.
        self.url_title_json.clear()



async def main():
    dataframe = pd.read_csv("request_DOI.csv")
    batch_size = 30
    out_file = "out.csv"
    crossref = CrossRef(dataframe, batch_size, out_file)
    await crossref.run()

if __name__ == "__main__":
    total_number = 25167
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    elapsed_time = end - start
    logger.warning(f"Run time: {elapsed_time:.2f}, for each costed ~ {elapsed_time/total_number:.4f}")