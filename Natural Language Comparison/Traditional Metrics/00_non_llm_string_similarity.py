from ragas.metrics.collections import NonLLMStringSimilarity, DistanceMeasure
import asyncio


async def main():
    scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)
    result = await scorer.ascore(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    print(f"NonLLM String Similarity Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())