from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolCall
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

# ToolCallAccuracy is deterministic; no LLM needed.


async def main():
    user_input = [
        HumanMessage(content="What's the weather like in New York right now?"),
        AIMessage(
            content="The current temperature in New York is 75°F and it's partly cloudy.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
        HumanMessage(content="Can you translate that to Celsius?"),
        AIMessage(
            content="Let me convert that to Celsius for you.",
            tool_calls=[
                ToolCall(
                    name="temperature_conversion", args={"temperature_fahrenheit": 75}
                )
            ],
        ),
    ]

    reference_tool_calls = [
        ToolCall(name="weather_check", args={"location": "New York"}),
        ToolCall(name="temperature_conversion", args={"temperature_fahrenheit": 75}),
    ]

    sample = MultiTurnSample(
        user_input=user_input,
        reference_tool_calls=reference_tool_calls,
    )

    scorer = ToolCallAccuracy()
    score = await scorer.multi_turn_ascore(sample)
    print(f"Tool Call Accuracy: {score}")


if __name__ == "__main__":
    asyncio.run(main())