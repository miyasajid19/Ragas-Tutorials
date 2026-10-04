from ragas.metrics.collections import ExactMatch
import asyncio


async def main():
    scorer = ExactMatch()
    result = await scorer.ascore(
        reference="Paris",
        response="India",
    )
    print(f"Exact Match Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())