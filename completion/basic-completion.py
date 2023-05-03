import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.9
    )

    return response.choices[0].text

prompt = input("Enter your prompt: ")
print(generate_text(prompt))
