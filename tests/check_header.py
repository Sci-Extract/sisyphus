from dotenv import load_dotenv, find_dotenv
import os
from openai import AsyncOpenAI
import asyncio

_ = load_dotenv(find_dotenv())

client = AsyncOpenAI(
)


async def main() -> None:
    chat_completion = await client.chat.completions.with_raw_response.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )

    print(chat_completion.headers)
    completion = chat_completion.parse()  # get the object that `chat.completions.create()` would have returned
    print(completion)


asyncio.run(main())