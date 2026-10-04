from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference

context_precision = NonLLMContextPrecisionWithReference()

sample = SingleTurnSample(
    retrieved_contexts=["The Eiffel Tower is located in Paris."],
    reference_contexts=["Paris is the capital of France.", "The Eiffel Tower is one of the most famous landmarks in Paris."]
)
async def main():
    result = await context_precision.single_turn_ascore(sample)
    print(f"Context Precision Score: {result}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())