from datasets import load_dataset
import argparse
import random
import re
import os
import pandas as pd
from tqdm import tqdm
import torch
import wandb
from sklearn.model_selection import train_test_split
from bert_score import score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

def get_dataset(data_split=":40%"):
    dataset = load_dataset("code_search_net", "python", split=f"train[{data_split}]")
    print(f"Dataset size: {len(dataset)}")
    return dataset

def remove_columns(dataset):
    data = dataset.to_pandas()
    data = data[['func_code_string', 'func_code_tokens']]
    print("Column Removed")

    return Dataset.from_pandas(data)

def remove_comment(dataset):
    data = dataset.to_pandas()

    single_line_comment_pattern = r"#.*"
    multi_line_comment_pattern = r'(?:(?:r|R)?("""|\'\'\')).*?\1'
    
    def remove_comments(code):
        code_no_multiline = re.sub(multi_line_comment_pattern, '', code, flags=re.DOTALL)
        code_no_comments = re.sub(single_line_comment_pattern, '', code_no_multiline)
        return code_no_comments.strip()
    
    data['cleaned_method'] = data['func_code_string'].apply(remove_comments)

    print("Comments, Doctring Removed")

    return Dataset.from_pandas(data)

def filter_dataset(dataset):
    # Remove Unnecessary Columns
    dataset = remove_columns(dataset)
    
    # Remove Test Method
    dataset = remove_test_method(dataset)

    # Remove Docstring or Comments(Multiple Line or Single Line)
    dataset = remove_comment(dataset)

    return dataset

def remove_test_method(dataset):
    data = dataset.to_pandas()

    pattern = r"^\s*def\s*test.*"

    filtered_data = data[~data['func_code_string'].str.match(pattern, na=False)]    
    removed_rows = len(data) - len(filtered_data)

    print(f"Number of 'test' functions removed: {removed_rows}")

    return Dataset.from_pandas(filtered_data)
