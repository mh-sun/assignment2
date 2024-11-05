from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments
)
import torch 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import re
import random


data_files = 'flatten_dataset.csv'
dataset = load_dataset("csv", split="train",data_files=data_files)  


#tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("./pretrained_codet5")
print(tokenizer.encode("<extra_id_0>"))
dataset = dataset.select_columns(['cleaned_method','condition_line'])


def mask_if_statements(examples):
    # Regular expression pattern to match 'if' conditions from 'if' to the colon
    pattern = re.compile(r"if\s+(.*?):")

    masked_code = []

    for code in examples['cleaned_method']:
        # Find all 'if' conditions in the code
        matches = list(pattern.finditer(code))

        # If no 'if' conditions are found, add the original code to the masked list
        if not matches:
            masked_code.append(code)
            continue

        # Randomly select one 'if' condition to mask
        selected_match = random.choice(matches)

        # Replace the selected 'if' condition with '<fill-in>'
        start, end = selected_match.span()
        code_with_mask = code[:start] + '<extra_id_0>' + code[end:]
        masked_code.append(code_with_mask)

    # Return the modified dataset
    return {
        "input_text": masked_code,  # Masked code snippet
        "target_text": examples['condition_line']  # original unmasked code snippet
    }

# Apply masking to create input-output pairs for T5
masked_dataset = dataset.map(mask_if_statements, batched=True)
# masked_dataset = dataset.select_columns(['cleaned_method', 'condition_line'])
print(masked_dataset)


# Remove rows that contain non-string values in "target_text"
def filter_invalid_texts(example):
    return isinstance(example['target_text'], str)

# Apply the filter to the dataset
filtered_dataset = masked_dataset.filter(filter_invalid_texts)

# Now define the tokenization function again
def tokenize_function(examples):
    # Extract input and target texts
    input_texts = examples["input_text"]
    target_texts = examples["target_text"]

    # Ensure all target_texts are strings (filtering should have handled this)
    assert all(isinstance(text, str) for text in target_texts), "All elements in target_texts must be strings."

    # Tokenizing input texts
    model_inputs = tokenizer(
        input_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

    # Tokenizing target texts for labels
    labels = tokenizer(
        target_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )["input_ids"]

    # Replace padding token id's with -100 for labels
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_list]
        for label_list in labels
    ]

    # Adding the labels to model_inputs
    model_inputs["labels"] = labels

    return model_inputs

# Tokenize the dataset using the corrected function
tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Split into train and test sets for training
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
model = AutoModelForSeq2SeqLM.from_pretrained("../pretrained_codet5")


# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=0.1,  # Reduced for testing
    weight_decay=0.01,
    save_total_limit=3,
)

# Initialize the Trainer with model, tokenizer, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

# Start Training
trainer.train()

# Save the trained model
trainer.save_model("fine_tuned_model")

#  Evaluation on test set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_model")

# Example test input data (a list of Python methods with masked parts)
test_inputs = [
    """def factorial(n):
    <extra_id_0>
        return 1
    else:
        return n * factorial(n-1)""",
    """"def word_frequency(text):
    words = text.split()
    frequency = {}
    for word in words:
        word = word.lower()
        <extra_id_0>
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency""", """x = 10 \n <extra_id_0> \n print('Greater than 5')\nelse:\n    print('5 or less')"""]

# Tokenize the test inputs
tokenized_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors="pt")


# Generate predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for inference
    output_sequences = model.generate(
        input_ids=tokenized_inputs.input_ids,
        attention_mask=tokenized_inputs.attention_mask,
        max_length=5000,  # Specify the maximum length of the generated output
        num_return_sequences=1,
        temperature=0.7  
    )

# Decode the generated sequences
for i, output in enumerate(output_sequences):
    decoded_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Test Input {i + 1}:")
    print(f"Original Input: {test_inputs[i]}")
    print(f"Model Prediction: {decoded_text}")
    print("-" * 50)