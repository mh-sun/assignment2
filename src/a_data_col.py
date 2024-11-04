import requests
import re

# Function to fetch repositories based on criteria
def fetch_repositories(query="language:python stars:>500"):
    url = f"https://api.github.com/search/repositories?q={query}"
    headers = {"Authorization": "token YOUR_GITHUB_TOKEN"}
    response = requests.get(url, headers=headers)
    return response.json()['items']

# Extract functions with if statements
def extract_functions_with_if(code_text):
    functions = re.findall(r'def .+:\n(?: {4}.+\n)+', code_text)
    if_functions = [func for func in functions if 'if ' in func]
    return if_functions

# Save functions to text file
def save_functions(functions, filename="python_code.txt"):
    with open(filename, "a") as f:
        for func in functions:
            f.write(func + "\n\n")

# Iterate over repositories and fetch code
repositories = fetch_repositories()
for repo in repositories:
    repo_url = repo['html_url']
    # Fetch repository content, parse and find functions with if statements

# Save final dataset
save_functions(all_functions, filename="data/python_code.txt")
