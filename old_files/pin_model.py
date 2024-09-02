import openai

# Set the OpenAI API key
openai.api_key = "your_openai_api_key"

# Pin to a specific model version
response = openai.Completion.create(  # type: ignore
    model="gpt-3.5-turbo",  # Pinning to a specific model
    prompt="Explain the importance of pinning models in production environments.",
    max_tokens=100,
    engine="text-davinci-003",  # Specify the engine
)

print(response.choices[0].text.strip())
