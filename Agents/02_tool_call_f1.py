from ragas.metrics import ToolCallF1
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolCall
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

# ToolCallF1 is deterministic; no LLM needed.


async def main():
    user_input = [
        HumanMessage(content="What's the weather like in Paris today?"),
        AIMessage(
            content="Let me check that for you.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "Paris"})],
        ),
        HumanMessage(content="And the UV index?"),
        AIMessage(
            content="Sure, here's the UV index for Paris.",
            tool_calls=[ToolCall(name="uv_index_lookup", args={"location": "Paris"})],
        ),
    ]

    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "Paris"}),
        ToolCall(name="uv_index_lookup", args={"location": "Paris"}),
    ]

    sample = MultiTurnSample(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )

    scorer = ToolCallF1()
    score = await scorer.multi_turn_ascore(sample)
    print(f"Tool Call F1: {score}")


if __name__ == "__main__":
    asyncio.run(main())