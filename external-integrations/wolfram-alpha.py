#
# Generate a Wolfram Alpha query from a question, use the Wolfram Alpha API to get the correct
# answer, then generate a complete sentence that responds to the original question.
#
import os
import openai
import wolframalpha

# Try
# "What planetary moons are larger than Mercury?"

openai.api_key = os.getenv("OPENAI_API_KEY")
wolfram_app_id = os.getenv("WOLFRAM_API_KEY")  

def gptCompletion(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.9
    )

    return response.choices[0].text

def wolframQuery(query):
    client = wolframalpha.Client(wolfram_app_id)
    res = client.query(query)
    #print(res)
    answer = next(res.results).text

    return answer

question = input('Question: ')

prompt = f"""Give me a simple keyword query that could be used by Wolfram Alpha to perform the calculation for the following question. Only give me the query, not the answer or an explanation.
Q: {question}
A:"""
query = gptCompletion(prompt)
#print(f"The Wolfram Alpha query is {query}")

answer = wolframQuery(query)
#print(f"The answer from Wolfram Alpha is {answer}")

prompt = f"""Q: {question}
A: {answer}
Q: Please rephrase the answer as a complete sentence that responds to the original question.
A:"""

finalAnswer = gptCompletion(prompt)

print("Answer: " + finalAnswer)