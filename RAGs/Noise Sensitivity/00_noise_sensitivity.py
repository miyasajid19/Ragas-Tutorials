from openai import AsyncOpenAI
from ragas.llms import llm_factory
from dotenv import load_dotenv
import os
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import NoiseSensitivity
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
sample = SingleTurnSample(
    user_input="Where is the Eiffel Tower located?",
    response="The Eiffel Tower is located in Paris.",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=["Paris is the capital of France."],
)

# Create metric

# Evaluate
async def main():
    scorer = NoiseSensitivity(llm=llm)
    result = await scorer.ascore(
        user_input="What is the Life Insurance Corporation of India (LIC) known for?",
        response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
        reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
        retrieved_contexts=[
            "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
            "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
            "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
            "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
        ]
    )
    print(f"Noise Sensitivity Score: {result.value}")

    scorer = NoiseSensitivity(llm=llm, mode="irrelevant")
    result = await scorer.ascore(
        user_input="What is the Life Insurance Corporation of India (LIC) known for?",
        response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
        reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
        retrieved_contexts=[
            "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
            "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
            "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
            "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
        ]
    )
    print(f"Noise Sensitivity (Irrelevant) Score: {result.value}")
if __name__ == "__main__":
    asyncio.run(main())