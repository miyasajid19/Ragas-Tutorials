from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field


class MyInput(BaseModel):
    question: str = Field(description="The question to answer")


class MyOutput(BaseModel):
    answer: str = Field(description="The answer to the question")


class MyPrompt(PydanticPrompt[MyInput, MyOutput]):
    instruction = "Answer the given question"
    input_model = MyInput
    output_model = MyOutput
    examples = [
        (
            MyInput(question="Who's building the opensource standard for LLM app evals?"),
            MyOutput(answer="Ragas"),
        )
    ]


if __name__ == "__main__":
    prompt = MyPrompt()
    print(f"Instruction: {prompt.instruction}")
    print(f"Input model: {prompt.input_model.__name__}")
    print(f"Output model: {prompt.output_model.__name__}")
    print(f"Examples: {len(prompt.examples)}")
    for i, (inp, out) in enumerate(prompt.examples):
        print(f"  [{i}] Q: {inp.question} -> A: {out.answer}")