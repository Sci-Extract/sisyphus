import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
from sisyphus.processor.parallel_processor import CompletionRequest

_ = load_dotenv(find_dotenv()) # load env


async def make_request():
    async with AsyncOpenAI() as client:
        request_maker = CompletionRequest(client, 5000, 80000, 3)
        request_json = [{"messages":[{"role": "user", "content": "say hello"}], "model": "gpt-3.5-turbo-1106"}]
        await request_maker.completion_helper_with_no_probe(iter(request_json), "test_result.jsonl")

asyncio.run(make_request())
