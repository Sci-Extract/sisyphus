import asyncio
import time
from sisyphus.patch import OpenAIEmbeddingThrottle, aembed_httpx_client
from langchain_openai import OpenAIEmbeddings


embed_o = OpenAIEmbeddings(http_async_client=aembed_httpx_client)
embed = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
async def main():
    start = time.perf_counter()
    texts = \
        [
            "hello, my name is soike",
            "how are you",
            "I want to be better",
            "wish that come true",
            "thanks!"
        ]
    await asyncio.gather(*[embed.aembed_query(doc) for doc in texts])
    elapsed = time.perf_counter() - start
    print(elapsed)
asyncio.run(main())