import asyncio
from typing import List

import requests
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import SemanticSimilarity


# --------------------------------------------------
# 1. Local Ollama embeddings (modern provider)
# --------------------------------------------------
class OllamaRagasEmbedding(BaseRagasEmbedding):
    """Minimal modern-ragas embeddings provider backed by Ollama's HTTP API."""

    def __init__(self, model: str = "embeddinggemma:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_text(self, text: str, **kwargs) -> List[float]:
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)


embeddings = OllamaRagasEmbedding(model="embeddinggemma:latest")


# --------------------------------------------------
# 2. Ragas Metric
# --------------------------------------------------

scorer = SemanticSimilarity(embeddings=embeddings)


# --------------------------------------------------
# 3. Evaluate
# --------------------------------------------------
async def main():
    result = await scorer.ascore(
        reference="The Eiffel Tower is located in Paris. It has a height of 1000ft.",
        response="The Eiffel Tower is located in Paris.",
    )
    print(f"Semantic Similarity Score: {result.value}")


if __name__ == "__main__":
    asyncio.run(main())