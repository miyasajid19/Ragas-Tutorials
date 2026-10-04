from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor, KeyphrasesExtractor
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)
from ragas.testset.transforms import apply_transforms, Parallel
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MINIMAX_API_KEY = os.getenv("AWS_API_KEY")
MINIMAX_API_BASE_URL = os.getenv("AWS_BASE_URL")
MINIMAX_MODEL = os.getenv("AWS_MODEL")

if not all([MINIMAX_API_KEY, MINIMAX_API_BASE_URL, MINIMAX_MODEL]):
    raise ValueError("Missing MiniMax environment variables")

# LangChain ChatOpenAI -> RagAS LangchainLLMWrapper (LLM-based extractors go through
# the legacy agenerate_prompt path; modern InstructorLLM lacks that method)
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=MINIMAX_MODEL,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_API_BASE_URL,
        temperature=0,
    )
)


async def main():
    sample_nodes = [
        Node(
            properties={
                "page_content": (
                    "Einstein's theory of relativity revolutionized our understanding "
                    "of space and time."
                )
            }
        ),
        Node(
            properties={
                "page_content": (
                    "Time dilation occurs when an object moves close to the speed of "
                    "light, causing time to pass slower relative to a stationary "
                    "observer."
                )
            }
        ),
    ]

    rel_builder = JaccardSimilarityBuilder(
        property_name="entities",
        key_name="PER",
        new_property_name="entity_jaccard_similarity",
    )

    kg = KnowledgeGraph(nodes=sample_nodes)

    transforms = [
        Parallel(KeyphrasesExtractor(llm=llm), NERExtractor(llm=llm)),
        rel_builder,
    ]

    try:
        apply_transforms(kg, transforms)
    except AttributeError as e:
        # Current NERExtractor returns a flat list while the docs' builder
        # expects a dict keyed by entity type. Skip the relationship step.
        print(f"[skip] apply_transforms skipped: {e}")

    print("First node properties after Parallel(KeyphrasesExtractor, NERExtractor):")
    print(kg.nodes[0].properties)


if __name__ == "__main__":
    asyncio.run(main())