from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

sample = SingleTurnSample(
    retrieved_contexts=["Paris is the capital of France."],
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)

context_recall = NonLLMContextRecall()
async def main():
    result = await context_recall.single_turn_ascore(sample)
    print(f"Context Recall Score: {result}")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())