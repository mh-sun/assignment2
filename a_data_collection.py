import os
import re
import csv
import subprocess
import pandas as pd

# Base GitHub URL and output CSV path
base_url = 'https://github.com/'
output_csv = 'generated.csv'

# Load repository data from CSV file
df = pd.read_csv('repo_links.csv')

# Initialize CSV file and write header once
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['original_method'])

# Regular expressions to identify Python methods and `if` conditions
method_pattern = r'def\s+\w+\s*\(.*?\):'  # Match function definitions
if_pattern = r'\bif\s*\(?.*?\):'           # Match `if` conditions

# Function to extract methods containing `if` statements from a given file
def extract_methods_with_if(file_path, methods_with_if):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

            in_method = False
            method_lines = []

            for line in lines:
                if re.match(method_pattern, line):
                    # If we're already in a method, finalize it before starting a new one
                    if in_method:
                        method_text = ''.join(method_lines)
                        if re.search(if_pattern, method_text):
                            methods_with_if.append((file_path, method_text.strip()))
                        method_lines = []

                    # Start of a new method
                    in_method = True

                if in_method:
                    method_lines.append(line)

                # End of method when indentation reduces (simple heuristic)
                if in_method and line.strip() == "":
                    in_method = False
                    method_text = ''.join(method_lines)
                    if re.search(if_pattern, method_text):
                        methods_with_if.append((file_path, method_text.strip()))
                    method_lines = []
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping...")

# Loop through repositories listed in the DataFrame
for i in range(1, len(df)):
    repo_url = base_url + df['name'][i] + '.git'
    repo_parts = df['name'][i].split('/')
    owner = repo_parts[0]
    repo_name = repo_parts[1]
    repo_dir = os.path.join('repos', owner, repo_name)
    print('The value of i is: ', i)

    # Step 1: Clone the repository if it hasn't been cloned already
    if not os.path.exists(repo_dir):
        print(f"Cloning the {repo_name} repository...")
        subprocess.run(['git', 'clone', repo_url, repo_dir])
    else:
        print(f"Repository {repo_name} already cloned.")

    # Step 2: List to hold methods with `if` statements for this repository
    methods_with_if = []

    # Loop through each Python file in the repository
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.py'):
                extract_methods_with_if(os.path.join(root, file), methods_with_if)

    # Step 3: Append methods with `if` conditions to the CSV file
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for file_path, method in methods_with_if:
            writer.writerow([method])

    if methods_with_if:
        print(f"Methods containing 'if' conditions from {repo_name} have been saved to {output_csv}")
    else:
        print(f"No methods with 'if' conditions found in {repo_name}.")
