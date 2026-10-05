from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

# User asks about the weather in New York City
user_message = HumanMessage(content="What's the weather like in New York City today?")

# AI decides to use a weather API tool to fetch the information
ai_initial_response = AIMessage(
    content="Let me check the current weather in New York City for you.",
    tool_calls=[ToolCall(name="WeatherAPI", args={"location": "New York City"})],
)

# Tool provides the weather information
tool_response = ToolMessage(
    content="It's sunny with a temperature of 75°F in New York City."
)

# AI delivers the final response to the user
ai_final_response = AIMessage(
    content="It's sunny and 75 degrees Fahrenheit in New York City today."
)

# Combine all messages into a list to represent the conversation
conversation = [
    user_message,
    ai_initial_response,
    tool_response,
    ai_final_response,
]

# Reference response for evaluation purposes
reference_response = "Provide the current weather in New York City to the user."

# Create the MultiTurnSample instance
sample = MultiTurnSample(
    user_input=conversation,
    reference=reference_response,
)

if __name__ == "__main__":
    print(sample.to_dict())