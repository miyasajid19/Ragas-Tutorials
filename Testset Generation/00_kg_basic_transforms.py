from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)
from ragas.testset.transforms import apply_transforms
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
                    "of space and time. It introduced the concept that time is not "
                    "absolute but can change depending on the observer's frame of "
                    "reference."
                )
            }
        ),
        Node(
            properties={
                "page_content": (
                    "Time dilation occurs when an object moves close to the speed of "
                    "light, causing time to pass slower relative to a stationary "
                    "observer. This phenomenon is a key prediction of Einstein's "
                    "special theory of relativity."
                )
            }
        ),
    ]

    print("Nodes:")
    print(sample_nodes)

    extractor = NERExtractor(llm=llm)
    output = [await extractor.extract(node) for node in sample_nodes]
    print("\nNER output for first node:")
    print(output[0])

    _ = [
        node.properties.update({key: val})
        for (key, val), node in zip(output, sample_nodes)
    ]
    print("\nFirst node properties after extraction:")
    print(sample_nodes[0].properties)

    kg = KnowledgeGraph(nodes=sample_nodes)
    rel_builder = JaccardSimilarityBuilder(
        property_name="entities",
        key_name="PER",
        new_property_name="entity_jaccard_similarity",
    )
    try:
        relationships = await rel_builder.transform(kg)
        print("\nRelationships:")
        print(relationships)
    except AttributeError as e:
        # Current NERExtractor returns a flat list of entities while the docs'
        # JaccardSimilarityBuilder expects a dict keyed by entity type (e.g.
        # {"PER": [...]}). The relationship step is skipped here for that
        # version mismatch; the KG itself is built and inspectable above.
        print(f"\n[skip] Relationship builder skipped: {e}")

    transforms = [extractor, rel_builder]
    try:
        apply_transforms(kg, transforms)
    except AttributeError as e:
        print(f"[skip] apply_transforms skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())