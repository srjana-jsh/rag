# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://test-chatgpt-flomoney.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    engine="finbot-gpt",
    messages=[],
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
)


# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://test-chatgpt-flomoney.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    engine="finbot-gpt",
    messages=[
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {"role": "user", "content": "hi how much money should i invest?"},
        {
            "role": "assistant",
            "content": "As an AI language model, I cannot provide financial advice. The amount of money you should invest depends on your financial goals, risk tolerance, and current financial situation. It is recommended to consult with a financial advisor who can help you create a personalized investment plan based on your individual needs and circumstances.",
        },
    ],
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
)
