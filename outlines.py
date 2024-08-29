from pydantic import BaseModel
from outlines import models, generate


# Define the desired output structure using Pydantic
class SentimentInfo(BaseModel):
    statement: str
    sentiment: str


# Initialize the model
model = models.transformers("microsoft/Phi-3-mini-4k-instruct")

# Set up the generator to create JSON output based on the SentimentInfo schema
generator = generate.json(model, SentimentInfo)

# Generate the JSON output for sentiment analysis
result = generator(
    "Analyze the sentiment of the statement 'The team's performance was lackluster, and morale is low.' "
    "and return the sentiment as either 'positive', 'negative', or 'neutral'."
)

# Print the generated result
print(result)
