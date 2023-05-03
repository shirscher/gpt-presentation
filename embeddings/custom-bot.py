#
# Shows how you can answer a question by searching a corpus of text for the most related entry.
# Use the file https://cdn.openai.com/API/examples/data/olympics_sections_text.csv
# This file already has the embeddings calculated for each entry.
#
import ast
import os
import openai
import numpy as np
import pandas as pd
from scipy import spatial  # for calculating vector similarities for search

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_embedding(prompt):
    response = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def strings_ranked_by_relatedness(
    df,
    related_embedding,
    top_n
):
    strings_and_relatednesses = [
        (row["content"], np.dot(np.array(related_embedding), np.array(ast.literal_eval(row["embedding"]))))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.9
    )

    return response.choices[0].text

df = pd.read_csv('./olympics_sections_text.csv')

question = input("Question: ")
question_embedding = generate_embedding(question)

print("Ranking articles by relatedness to question...")
ranked = strings_ranked_by_relatedness(df, question_embedding, 2)

prompt = f"""Use the below articles on the 2020 Summer Olympics to answer the subsequent question. If the answer cannot be found, write "I don't know."

Article 1:
\"\"\"
{ranked[0][0]}
\"\"\"

Article 2:
\"\"\"
{ranked[1][0]}
\"\"\"

Q: {question}
A:"""

print("Generating answer...")
#print("Prompt: " + prompt)
answer = generate_text(prompt)

print("Answer: " + answer)