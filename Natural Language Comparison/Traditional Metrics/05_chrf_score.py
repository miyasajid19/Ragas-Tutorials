from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ChrfScore
import asyncio


async def main():
    sample = SingleTurnSample(
        response="The Eiffel Tower is located in India.",
        reference="The Eiffel Tower is located in Paris.",
    )
    scorer = ChrfScore()
    score = await scorer.single_turn_ascore(sample)
    print(f"CHRF Score: {score}")


if __name__ == "__main__":
    asyncio.run(main())