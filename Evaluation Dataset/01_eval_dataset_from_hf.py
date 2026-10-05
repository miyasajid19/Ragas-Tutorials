from datasets import load_dataset
from ragas import EvaluationDataset

dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")

eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

if __name__ == "__main__":
    print(f"Total samples: {len(eval_dataset.samples)}")
    if eval_dataset.samples:
        first = eval_dataset.samples[0]
        print(f"First user_input: {first.user_input[:80]}...")
        print(f"First reference: {first.reference[:80] if first.reference else None}...")