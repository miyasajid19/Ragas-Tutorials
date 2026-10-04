from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("AWS_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("AWS_BASE_URL")
MINIMAX_MODEL = os.getenv("AWS_MODEL")

if not all([MINIMAX_API_KEY, MINIMAX_API_BASE_URL, MINIMAX_MODEL]):
    raise ValueError("Missing MiniMax environment variables")

# Modern llm_factory (DiscreteMetric uses the InstructorLLM agenerate path, not agenerate_prompt)
client = AsyncOpenAI(
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_API_BASE_URL,
)

evaluator_llm = llm_factory(
    MINIMAX_MODEL,
    client=client,
)


async def main():
    clarity_metric = DiscreteMetric(
        name="clarity",
        allowed_values=list(range(0, 11)),  # 0 to 10
        prompt="""Rate the clarity of the response on a scale of 0-10.
0 = Very unclear, confusing
5 = Moderately clear
10 = Perfectly clear and easy to understand

Response: {response}

Respond with only the number (0-10).""",
    )

    response = (
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn from data."
    )

    result = await clarity_metric.ascore(response=response, llm=evaluator_llm)
    print(f"Clarity Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())