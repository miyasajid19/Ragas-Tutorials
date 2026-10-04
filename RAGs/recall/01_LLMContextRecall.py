from openai import AsyncOpenAI
from ragas.llms import llm_factory
from dotenv import load_dotenv
import os
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall
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

context_recall = LLMContextRecall(llm=llm)
async def main():
    result = await context_recall.single_turn_ascore(sample)
    print(f"Context Recall Score: {result.value}")

if __name__ == "__main__":
    asyncio.run(main())
    
# this does not work