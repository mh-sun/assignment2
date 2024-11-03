import pandas as pd
import re
import tokenize
import io

def filter_dataset(input_path, output_path):
    data = pd.read_csv(input_path)
    # data = data.sample(10, random_state=42)
    print(f"Original Data Length: {len(data)}")
    
    pattern = r"^\s*def\s*test.*"

    filtered_data = data[~data['original_method'].str.match(pattern, na=False)]
    print(f"Filtered Data Length: {len(filtered_data)}")
    
    removed_rows = len(data) - len(filtered_data)
    print(f"Number of 'test' functions removed: {removed_rows}")
    
    filtered_data.to_csv(output_path, index=False)
    print("Filtered dataset saved as 'filtered_dataset.csv'")

def prepare_dataset(input_path, output_path):
    data = pd.read_csv(input_path)

    data = remove_comment(data)
    data = remove_multiple_newline(data)

    data.to_csv(output_path, index=False)
    print(f"Prepared dataset saved as '{output_path}'")

def remove_multiple_newline(data):
    multiple_newlines_pattern = r'\n+'
    
    def remove_extra_newlines(code):
        return re.sub(multiple_newlines_pattern, '\n', code).strip()
    
    data['cleaned_method'] = data['cleaned_method'].apply(remove_extra_newlines)
    
    print(f"Data Length after removing extra newlines: {len(data)}")
    return data


def remove_comment(data):
    single_line_comment_pattern = r"#.*"
    multi_line_comment_pattern = r'(?:(?:r|R)?("""|\'\'\')).*?\1'
    
    def remove_comments(code):
        code_no_multiline = re.sub(multi_line_comment_pattern, '', code, flags=re.DOTALL)
        code_no_comments = re.sub(single_line_comment_pattern, '', code_no_multiline)
        return code_no_comments.strip()
    
    data['cleaned_method'] = data['original_method'].apply(remove_comments)

    print(f"Data Length after comment removal: {len(data)}")
    return data

def flatten_dataset(input_path, output_path):
    data = pd.read_csv(input_path)
    print(f"Original Data Length: {len(data)}")

    # Flatten methods
    def flatten_method(code):
        flattened = code.strip()  
        flattened = flattened.replace(' ', ' <SPACE> ')
        flattened = flattened.replace('\n', ' <NEWLINE> ') 
        flattened = flattened.replace('\t', ' <TAB> ')  
        
        flattened = flattened.replace(' <TAB> ', ' <SPACE> ' * 4)  
        return flattened

    data['flattened_method'] = data['masked_method'].apply(flatten_method)

    data.to_csv(output_path, index=False)
    print(f"Flattened dataset saved as '{output_path}'")


def cut_condition(input_path, output_path):
    data = pd.read_csv(input_path)
    
    condition_pattern = r'\n+?\s*?if.*?\:\s*?\n'

    def process_method(code):
        match = re.search(condition_pattern, code)
        if match:
            condition_line = match.group().strip() 
            masked_method = code.replace(condition_line, '[MASK]')

            return masked_method, condition_line
        return code, "" 
    
    data[['masked_method', 'condition_line']] = data['cleaned_method'].apply(process_method).apply(pd.Series)

    def character_count(input_string):
        return len(input_string)
    
    data['masked_method_ch_count'] = data['masked_method'].apply(character_count)
    data['condition_line_ch_count'] = data['condition_line'].apply(character_count)

    data.to_csv(output_path, index=False)
    print(f"Cut condition dataset saved as '{output_path}'")


if __name__ == "__main__":
    dataset = "dataset/generated.csv"
    filtered_dataset = "dataset/filtered_dataset.csv"
    processed_dataset = "dataset/python_methods.csv"
    flattened_dataset = "dataset/flatten_dataset.csv"
    cut_condition_dataset = "dataset/cut_condition_dataset.csv"

    filter_dataset(dataset, filtered_dataset)
    prepare_dataset(filtered_dataset, processed_dataset)
    cut_condition(processed_dataset, cut_condition_dataset)
    flatten_dataset(cut_condition_dataset, flattened_dataset)
    
