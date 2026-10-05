from ragas import SingleTurnSample, EvaluationDataset

# Sample 1
sample1 = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany."],
    response="The capital of Germany is Berlin.",
    reference="Berlin",
)

# Sample 2
sample2 = SingleTurnSample(
    user_input="Who wrote 'Pride and Prejudice'?",
    retrieved_contexts=["'Pride and Prejudice' is a novel by Jane Austen."],
    response="'Pride and Prejudice' was written by Jane Austen.",
    reference="Jane Austen",
)

# Sample 3
sample3 = SingleTurnSample(
    user_input="What's the chemical formula for water?",
    retrieved_contexts=["Water has the chemical formula H2O."],
    response="The chemical formula for water is H2O.",
    reference="H2O",
)

dataset = EvaluationDataset(samples=[sample1, sample2, sample3])

if __name__ == "__main__":
    print(f"Total samples: {len(dataset.samples)}")
    for i, sample in enumerate(dataset.samples, 1):
        print(f"  [{i}] Q: {sample.user_input} -> A: {sample.response}")