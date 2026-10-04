from ragas.metrics.collections import RougeScore
import asyncio


async def main():
    scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")
    result = await scorer.ascore(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    print(f"ROUGE Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())