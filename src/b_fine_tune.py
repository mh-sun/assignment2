import argparse
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments
)
import torch 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import re
import random
from utils import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model-codet5")
model = AutoModelForSeq2SeqLM.from_pretrained("model-codet5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def mask_if_statements(examples):
    pattern = re.compile(r"if\s+(.*?):")
    masked_code = []
    target_conditions = []

    for code in examples['cleaned_method']:
        matches = list(pattern.finditer(code))

        # If no 'if' conditions are found, add the original code and empty target
        if not matches:
            masked_code.append(code)
            target_conditions.append("") 
            continue

        # Randomly select one 'if' condition to mask
        selected_match = random.choice(matches)
        start, end = selected_match.span()
        code_with_mask = code[:start] + '<extra_id_0>' + code[end:]

        masked_code.append(code_with_mask)
        target_conditions.append("if " + selected_match.group(1) + ":")

    # Return the modified dataset
    return {
        "input_text": masked_code,
        "target_text": target_conditions
    }

def mask_if_condition(dataset):
    dataset = dataset.map(mask_if_statements, batched=True)
    return dataset

def tokenize_function(examples):
    input_texts = examples["input_text"]
    target_texts = examples["target_text"]

    assert all(isinstance(text, str) for text in target_texts), "All elements in target_texts must be strings."

    model_inputs = tokenizer(
        input_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=64
    )

    labels = tokenizer(
        target_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=64
    )["input_ids"]

    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_list]
        for label_list in labels
    ]

    model_inputs["labels"] = labels

    return model_inputs

def generate_prediction(path, test_dataset, args):
    out = args.out
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Define device

    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    test_df = test_dataset.to_pandas()
    test_inputs = test_df['input_text'].tolist()
    if_gt = test_df['target_text'].tolist()

    tokenized_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    
    model.eval()
    results = []
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_length=128,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True
        )

    # Decode the generated sequences and calculate scores
    for i, (output, scores) in enumerate(zip(output_sequences.sequences, output_sequences.scores)):
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Calculate token probabilities
        token_probs = [F.softmax(score, dim=-1) for score in scores]
        
        # Extract probabilities of selected tokens
        selected_probs = [
            token_probs[j][token_id].item()  # Use only token_id for indexing
            for j, token_id in enumerate(output[1:])  # Skip initial token in output
        ]

        # Calculate average prediction score
        if selected_probs:
            prediction_score = sum(selected_probs) / len(selected_probs) * 100
        else:
            prediction_score = 0.0

        # Check if the prediction matches the ground truth
        is_correct = decoded_text.strip() == if_gt[i].strip()

        # Append the result in the specified format
        results.append({
            "input": test_inputs[i],
            "prediction_status": is_correct,
            "ground_truth_if": if_gt[i],
            "prediction_if": decoded_text,
            "prediction_score": round(prediction_score, 2)
        })

    # Convert results to a DataFrame and save as a CSV
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"Results saved to {out}")


def train_model(dataset, args):
    data = dataset.to_pandas()
    data = data[['input_text', 'target_text']]
    dataset = Dataset.from_pandas(data)

    epoch = args.epoch
    lr = args.learning_rate

    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    temp = train_test_split['test']

    train_test_split = temp.train_test_split(test_size=0.5)
    valid_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    valid_dataset = valid_dataset.shuffle(seed=42).select(range(min(1000, len(valid_dataset))))
    test_dataset = test_dataset.shuffle(seed=42).select(range(min(1000, len(test_dataset))))

    print(train_dataset)
    print(valid_dataset)
    print(test_dataset)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
    valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
    # test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=epoch, 
        weight_decay=0.01,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("fine_tuned_model")

    generate_prediction("fine_tuned_model", test_dataset, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Code-T5 Model on Code Search Net Dataset")

    parser.add_argument('-e', '--epoch', type=int, default=1, help="Finetune Parameter: Epoch")
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help="Finetune Parameter: Learning Rate")
    parser.add_argument('-o', '--out', type=str, default="dataset_v2/generated/evaluation_finetune_results.csv", help="Finetune Parameter: Test Result Path")

    args = parser.parse_args()

    dataset = get_dataset("10%:20%")
    dataset = filter_dataset(dataset)
    dataset = mask_if_condition(dataset)

    train_model(dataset, args)

    wandb.finish()