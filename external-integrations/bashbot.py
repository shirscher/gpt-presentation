#
# Generates bash scripts to perform a user action.
#
import openai
import os

# Try "find all files in the ./gpt-presentation folder that have the word 'wolfram' in them"

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.9
    )

    return response.choices[0].text

action = input("Action: ")
prompt = F"""
You are a bash shell scripting bot. You generate bash scripts that perform a user action. Output just the bash script, not an explanation or the output of the script.

User Action: {action}
Bash script:"""
script = generate_text(prompt)

# Write the script out to a .sh file and execute it
with open("script.sh", "w") as f:
    f.write(script)

#os.system("chmod +x script.sh")
#os.system("./script.sh")
