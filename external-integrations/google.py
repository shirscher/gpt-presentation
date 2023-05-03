#
# Answer a question by searching the web for the answer, summarizing the top 3 results, and then
# generating a complete sentence that responds to the original question.
#
import os
import openai
from serpapi import GoogleSearch
import trafilatura

# Try: Who are the villians in the movie Spider-Man: No Way Home?
# GPT will know the movie is planned but not released yet, so it will not know the answer.

openai.api_key = os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")  

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.9
    )
    
    return response.choices[0].text

def extract_web_text(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded)

def summarize_page(title, url):
    text = extract_web_text(result['link'])
    prompt = f"""Summarize the following web page
    Title: {result['title']}
    Content: {text}"""
    
    summary = generate_text(prompt)
    return f"""Title: {result['title']}\nSummary: {summary}\n"""

question = input("Question: ")

search = GoogleSearch({
    "q": question,
    "api_key": serp_api_key
    })

result = search.get_dict()

summaries = ""
for result in result["organic_results"].take(3):
    #print(f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n")
    summary = summarize_page(result['title'], result['link'])
    summaries += summary + "\n"

prompt = f"""You are a helpful assistant. Answer the question using the context below.

Context:
{summaries}

Question: {question}
Answer:"""

answer = generate_text(prompt)
print(answer)