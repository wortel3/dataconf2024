from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from openai import OpenAI
import os

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


client = OpenAI()

# Obtain a chat completion response from GPT-3.5-turbo for the Olympic scenario
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant who provides information about teams in the Olympic Village.",
        },
        {"role": "user", "content": "How is Team A feeling after their match?"},
    ],
)

# Get the actual output
actual_output = response.choices[0].message.content or ""
print("Response from GPT-3.5 Turbo: ", actual_output)


# Check the relevancy of the GPT-3.5 answer based on a threshold of 0.8
def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.8)
    test_case = LLMTestCase(
        input="How is Team A feeling after their match?",
        # Actual output of your LLM application
        actual_output=actual_output,
        # Expected output (this is what you think the model should ideally output)
        expected_output="Team A is feeling frustrated after losing their match. The players are disappointed, and morale is low.",
    )
    assert_test(test_case, [answer_relevancy_metric])


# Run the test
test_answer_relevancy()
