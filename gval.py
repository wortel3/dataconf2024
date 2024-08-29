from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai import OpenAI

# Set up the OpenAI client (replace with actual API key)
client = OpenAI()

# Define the prompt for the LLM
prompt = "Analyze the sentiment of the statement: 'The performance of Team A in the Olympics was outstanding, boosting the team's morale significantly.' Return either 'positive', 'negative', or 'neutral'."

# Obtain the LLM's output (this would be an actual API call in practice)
response = client.completions.create(
    model="gpt-3.5-turbo", prompt=prompt, max_tokens=10
)

# Extract the actual output from the LLM's response
actual_output = response.choices[0].text.strip()

# Define the expected output for this test case
expected_output = "positive"

# Create the correctness metric using GEval
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'.",
        "Heavily penalize omission of key details.",
        "Vague language or contradicting opinions are acceptable as long as the core facts are correct.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

# Define the test case using LLMTestCase
test_case = LLMTestCase(
    input=prompt,
    actual_output=actual_output,
    expected_output=expected_output,
    metrics=[correctness_metric],
)

# Run the test and print the result
result = correctness_metric.evaluate(test_case)
print(result)
