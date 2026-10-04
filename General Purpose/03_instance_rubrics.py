from ragas.metrics import InstanceRubrics
from ragas.llms import LangchainLLMWrapper
from ragas.evaluation import evaluate, EvaluationDataset
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("AWS_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("AWS_BASE_URL")
MINIMAX_MODEL = os.getenv("AWS_MODEL")

if not all([MINIMAX_API_KEY, MINIMAX_API_BASE_URL, MINIMAX_MODEL]):
    raise ValueError("Missing MiniMax environment variables")

# LangChain ChatOpenAI -> RagAS LangchainLLMWrapper (legacy metrics need langchain-style agenerate_prompt)
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=MINIMAX_MODEL,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_API_BASE_URL,
        temperature=0,
    )
)


async def main():
    dataset = [
        # Relevance to Query
        {
            "user_query": "How do I handle exceptions in Python?",
            "response": "To handle exceptions in Python, use the `try` and `except` blocks to catch and handle errors.",
            "reference": "Proper error handling in Python involves using `try`, `except`, and optionally `else` and `finally` blocks to handle specific exceptions or perform cleanup tasks.",
            "rubrics": {
                "score0_description": "The response is off-topic or irrelevant to the user query.",
                "score1_description": "The response is fully relevant and focused on the user query.",
            },
        },
        # Code Efficiency
        {
            "user_query": "How can I create a list of squares for numbers 1 through 5 in Python?",
            "response": """
                # Using a for loop
                squares = []
                for i in range(1, 6):
                    squares.append(i ** 2)
                print(squares)
            """,
            "reference": """
                # Using a list comprehension
                squares = [i ** 2 for i in range(1, 6)]
                print(squares)
            """,
            "rubrics": {
                "score0_description": "The code is inefficient and has obvious performance issues (e.g., unnecessary loops or redundant calculations).",
                "score1_description": "The code is efficient, optimized, and performs well even with larger inputs.",
            },
        },
    ]

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[InstanceRubrics(llm=evaluator_llm)],
        llm=evaluator_llm,
    )

    print(f"Instance Rubrics: {result}")


if __name__ == "__main__":
    asyncio.run(main())