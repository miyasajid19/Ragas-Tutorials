from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import SummaryScore
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("AWS_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("AWS_BASE_URL")
MINIMAX_MODEL = os.getenv("AWS_MODEL")

if not all([MINIMAX_API_KEY, MINIMAX_API_BASE_URL, MINIMAX_MODEL]):
    raise ValueError("Missing MiniMax environment variables")

# MiniMax OpenAI-compatible client
client = AsyncOpenAI(
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_API_BASE_URL,
)

# RAGAS LLM
llm = llm_factory(
    MINIMAX_MODEL,
    client=client,
)


async def main():
    scorer = SummaryScore(llm=llm)
    result = await scorer.ascore(
        reference_contexts=[
            "A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."
        ],
        response="A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders.",
    )
    print(f"Summary Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())