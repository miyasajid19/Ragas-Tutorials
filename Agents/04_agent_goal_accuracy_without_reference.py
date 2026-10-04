from ragas.metrics import AgentGoalAccuracyWithoutReference
from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.messages import HumanMessage, AIMessage, ToolCall, ToolMessage
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("AWS_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("AWS_BASE_URL")
MINIMAX_MODEL = os.getenv("AWS_MODEL")

if not all([MINIMAX_API_KEY, MINIMAX_API_BASE_URL, MINIMAX_MODEL]):
    raise ValueError("Missing MiniMax environment variables")

# LangChain ChatOpenAI -> RagAS LangchainLLMWrapper
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=MINIMAX_MODEL,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_API_BASE_URL,
        temperature=0,
    )
)


async def main():
    user_input = [
        HumanMessage(
            content="Hey, book a table at the nearest best Chinese restaurant for 8:00pm"
        ),
        AIMessage(
            content="Sure, let me find the best options for you.",
            tool_calls=[
                ToolCall(
                    name="restaurant_search",
                    args={"cuisine": "Chinese", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(
            content="Found a few options: 1. Golden Dragon, 2. Jade Palace"
        ),
        AIMessage(
            content="I found some great options: Golden Dragon and Jade Palace. Which one would you prefer?"
        ),
        HumanMessage(content="Let's go with Golden Dragon."),
        AIMessage(
            content="Great choice! I'll book a table for 8:00pm at Golden Dragon.",
            tool_calls=[
                ToolCall(
                    name="restaurant_book",
                    args={"name": "Golden Dragon", "time": "8:00pm"},
                )
            ],
        ),
        ToolMessage(content="Table booked at Golden Dragon for 8:00pm."),
        AIMessage(
            content="Your table at Golden Dragon is booked for 8:00pm. Enjoy your meal!"
        ),
        HumanMessage(content="thanks"),
    ]

    sample = MultiTurnSample(
        user_input=user_input,
    )

    scorer = AgentGoalAccuracyWithoutReference(llm=evaluator_llm)
    score = await scorer.multi_turn_ascore(sample)
    print(f"Agent Goal Accuracy (without reference): {score}")


if __name__ == "__main__":
    asyncio.run(main())