import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure using Pydantic
class SentimentInfo(BaseModel):
    statement: str
    sentiment: str


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data (sentiment analysis) from natural language
sentiment_info = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=SentimentInfo,
    messages=[
        {
            "role": "user",
            "content": "The team's performance was lackluster, and morale is low.",
        }
    ],
    prompt="""
Please analyze the sentiment of the statement and return it in JSON format.
The sentiment should be either "positive", "negative", or "neutral".
""",
)

# Print the extracted values
print("Statement:", sentiment_info.statement)
# > The team's performance was lackluster, and morale is low.
print("Sentiment:", sentiment_info.sentiment)
# > negative
