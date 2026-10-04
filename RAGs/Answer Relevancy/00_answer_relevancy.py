from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import AnswerRelevancy


# --------------------------------------------------
# 1. Ollama LLM
# --------------------------------------------------

ollama_llm = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0,
)

llm = LangchainLLMWrapper(ollama_llm)


# --------------------------------------------------
# 2. Ollama Embeddings
# --------------------------------------------------

ollama_embeddings = OllamaEmbeddings(
    model="embeddinggemma:latest",
)

embeddings = LangchainEmbeddingsWrapper(
    ollama_embeddings
)


# --------------------------------------------------
# 3. Ragas Metric
# --------------------------------------------------

scorer = AnswerRelevancy(
    llm=llm,
    embeddings=embeddings,
)


# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------
async def main():
    result = await scorer.ascore(
    user_input="When was the first Super Bowl?",
    response="The first Super Bowl was held on January 15, 1967."
)

    print(f"Answer Relevancy Score: {result.value}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())