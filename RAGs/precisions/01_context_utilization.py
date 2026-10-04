from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextUtilization
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("MINIMAX_BASE_URL")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL")

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

# Metric
scorer = ContextUtilization(llm=llm)
# The ContextUtilization metric evaluates whether retrieved contexts are useful by comparing each context against the generated response. Use this when you don't have a reference answer but have the response that was generated.

async def main():
    result = await scorer.ascore(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Eiffel Tower is located in Paris.",
            "The Brandenburg Gate is located in Berlin.",
        ],
    )

    print(f"Context Precision Score: {result.value}")

another_result = scorer.score(
    user_input="What is the capital of France?",
    response="The capital of France is Paris.",
    retrieved_contexts=[
        "The capital of Germany is Berlin.",
        "The capital of France is Paris.",
    ],
)
print(f"Context Precision Score (sync): {another_result.value}")
if __name__ == "__main__":
    asyncio.run(main())