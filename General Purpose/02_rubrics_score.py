from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import RubricsScore
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
        response="The Earth is flat and does not orbit the Sun.",
        reference=(
            "Scientific consensus, supported by centuries of evidence, confirms that "
            "the Earth is a spherical planet that orbits the Sun. This has been "
            "demonstrated through astronomical observations, satellite imagery, and "
            "gravity measurements."
        ),
    )

    rubrics = {
        "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
        "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
        "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
        "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
        "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions.",
    }

    scorer = RubricsScore(rubrics=rubrics, llm=evaluator_llm)
    score = await scorer.single_turn_ascore(sample)
    print(f"Rubric Score: {score}")


if __name__ == "__main__":
    asyncio.run(main())