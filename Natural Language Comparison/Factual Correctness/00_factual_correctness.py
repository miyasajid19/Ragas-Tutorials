from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness
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
    # Default mode is "f1"
    scorer = FactualCorrectness(llm=llm)
    result = await scorer.ascore(
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris. It has a height of 1000ft.",
    )
    print(f"Factual Correctness Score (f1): {result.value}")


if __name__ == "__main__":
    asyncio.run(main())