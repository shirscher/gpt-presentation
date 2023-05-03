import openai
import os
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_embedding(prompt):
    response = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )

    return response['data'][0]['embedding']

prompt = input("Enter your prompt: ")
promptEmbedding = generate_embedding(prompt)

#print(promptEmbedding)
#exit()

positiveEmbedding = generate_embedding("A positive statement")
negativeEmbedding = generate_embedding("A negative statement")

similarityToPositive = np.dot(promptEmbedding, positiveEmbedding) / (np.linalg.norm(promptEmbedding) * np.linalg.norm(positiveEmbedding))
similarityToNegative = np.dot(promptEmbedding, negativeEmbedding) / (np.linalg.norm(promptEmbedding) * np.linalg.norm(negativeEmbedding))

if similarityToPositive > similarityToNegative:
    print("Your prompt is positive")
else:
    print("Your prompt is negative")
