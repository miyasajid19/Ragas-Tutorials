from ragas.metrics.collections import StringPresence
import asyncio


async def main():
    scorer = StringPresence()
    result = await scorer.ascore(
        reference="Eiffel Tower",
        response="The Eiffel Tower is located in India.",
    )
    print(f"String Presence Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())