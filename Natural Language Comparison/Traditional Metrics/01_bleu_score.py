from ragas.metrics.collections import BleuScore
import asyncio


async def main():
    scorer = BleuScore()
    result = await scorer.ascore(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    print(f"BLEU Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())