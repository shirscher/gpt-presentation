#
# Basic LangChain example that uses the Search and Calculate tools to answer a question.
#
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

# Env variables OPENAI_API_KEY and SERPAPI_API_KEY must be set

# Try
# "What is Tesla's market capitalization as of March 3rd 2023, how many times bigger is this than Ford's market cap on the same day?"
question = input('Question: ')

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# This would use the react-docstore agent instead, which uses a few-shot prompt that more closely resembles the original ReAct paper.
# Doesn't work with the specified tools, though. So some tweaks would be required.
#agent = initialize_agent(tools, llm, agent="react-docstore", verbose=True) 

agent.run(question)
