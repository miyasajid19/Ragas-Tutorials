from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMSQLEquivalence
from ragas.llms import LangchainLLMWrapper
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
    sample = SingleTurnSample(
        response="""
            SELECT p.product_name, SUM(oi.quantity) AS total_quantity
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            GROUP BY p.product_name;
        """,
        reference="""
            SELECT products.product_name, SUM(order_items.quantity) AS total_quantity
            FROM order_items
            INNER JOIN products ON order_items.product_id = products.product_id
            GROUP BY products.product_name;
        """,
        reference_contexts=[
            """
            Table order_items:
            - order_item_id: INT
            - order_id: INT
            - product_id: INT
            - quantity: INT
            """,
            """
            Table products:
            - product_id: INT
            - product_name: VARCHAR
            - price: DECIMAL
            """,
        ],
    )

    scorer = LLMSQLEquivalence()
    scorer.llm = evaluator_llm
    result = await scorer.single_turn_ascore(sample)
    print(f"SQL Semantic Equivalence: {result}")


if __name__ == "__main__":
    asyncio.run(main())