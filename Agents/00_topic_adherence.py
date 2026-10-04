from ragas.metrics import TopicAdherenceScore
from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("AWS_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("AWS_BASE_URL")
MINIMAX_MODEL = os.getenv("AWS_MODEL")

if not all([MINIMAX_API_KEY, MINIMAX_API_BASE_URL, MINIMAX_MODEL]):
    raise ValueError("Missing MiniMax environment variables")

# LangChain ChatOpenAI -> RagAS LangchainLLMWrapper (legacy metrics need langchain-style agenerate_prompt)
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
            content="Can you provide me with details about Einstein's theory of relativity?"
        ),
        AIMessage(
            content="Sure, let me retrieve the relevant information for you.",
            tool_calls=[
                ToolCall(
                    name="document_search",
                    args={"query": "Einstein's theory of relativity"},
                )
            ],
        ),
        ToolMessage(
            content="Found relevant documents: 1. Relativity: The Special and the General Theory, 2. General Theory of Relativity by A. Einstein."
        ),
        AIMessage(
            content="I found some documents on Einstein's theory of relativity. Which one would you like to know more about: 'Relativity: The Special and the General Theory' or 'General Theory of Relativity by A. Einstein'?"
        ),
        HumanMessage(content="Tell me about the 'General Theory of Relativity'."),
        AIMessage(
            content="Got it! Let me fetch more details from 'General Theory of Relativity by A. Einstein'.",
            tool_calls=[
                ToolCall(
                    name="document_retrieve",
                    args={"document": "General Theory of Relativity by A. Einstein"},
                )
            ],
        ),
        ToolMessage(
            content="The document discusses how gravity affects the fabric of spacetime, describing the relationship between mass and spacetime curvature."
        ),
        AIMessage(
            content="The 'General Theory of Relativity' explains how gravity affects the fabric of spacetime and the relationship between mass and spacetime curvature. Would you like more details or a specific explanation?"
        ),
        HumanMessage(
            content="No, that's perfect. By the way, do you know any good recipes for a chocolate cake?"
        ),
        AIMessage(
            content="Sure! Let me find a simple and delicious recipe for a chocolate cake.",
            tool_calls=[
                ToolCall(name="recipe_search", args={"query": "chocolate cake recipe"})
            ],
        ),
        ToolMessage(
            content="Here's a popular recipe for a chocolate cake: Ingredients include flour, sugar, cocoa powder, eggs, milk, and butter. Instructions: Mix dry ingredients, add wet ingredients, and bake at 350°F for 30-35 minutes."
        ),
        AIMessage(
            content="I found a great recipe for chocolate cake! Would you like the full details, or is that summary enough?"
        ),
    ]

    sample = MultiTurnSample(
        user_input=user_input,
        reference_topics=["religon"],
    )

    scorer = TopicAdherenceScore(llm=evaluator_llm, mode="precision")
    score = await scorer.multi_turn_ascore(sample)
    print(f"Topic Adherence (precision): {score}")


if __name__ == "__main__":
    asyncio.run(main())