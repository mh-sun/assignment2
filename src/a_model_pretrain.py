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
from utils import *

wandb.init(project="code-t5-pretraining")

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def mask_dataset(dataset, mask_rate=0.15, seed=42):
    data = dataset.to_pandas()
    data[['masked_func', 'masked_tokens']] = data['func_code_tokens'].apply(
        lambda tokens: pd.Series(mask_text(tokens, mask_rate))
    )
    return Dataset.from_pandas(data)

def mask_text(tokenList, mask_rate):
    tokens = tokenList.copy()
    num_to_mask = max(1, int(len(tokens) * mask_rate))
    mask_indices = random.sample(range(len(tokens)), num_to_mask)

    mask_indices.sort()

    mask_tokens = []
    for i in mask_indices:
        mask_tokens.append(tokens[i])
        tokens[i] = "<extra_id_0>"

    return " ".join(tokens), "###".join(mask_tokens)

def pretrain_model(dataset, args):
    epoch = args.epoch

    data = dataset.to_pandas()
    data = data[['masked_func', 'masked_tokens']]
    dataset = Dataset.from_pandas(data)

    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    temp = train_test_split['test']

    train_test_split = temp.train_test_split(test_size=0.5)
    valid_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    def tokenize_function(examples):
        inputs = tokenizer(
            examples['masked_func'], return_tensors="pt", padding="max_length", truncation=True, max_length=128 
        ).to(device)
        labels = tokenizer(
            examples['masked_tokens'], return_tensors="pt", padding="max_length", truncation=True, max_length=128
        ).to(device)
        return {
            "input_ids": inputs.input_ids,
            "labels": labels.input_ids
        }
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['masked_func', 'masked_tokens'])
    valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=['masked_func', 'masked_tokens'])

    print(train_dataset)
    print(valid_dataset)
    print(test_dataset)

    training_args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs=epoch,
        evaluation_strategy="steps", 
        eval_steps=100,              
        save_strategy="steps",       
        save_steps=500,               
        learning_rate=1e-5,
        logging_steps=10,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        report_to="wandb", 
        logging_dir='logs'
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
    )

    def evaluate_model(model, td):
        test_df = td.to_pandas()
        model.eval()
        predictions = []

        refs = []
        preds = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            masked_text = row['masked_func']
            masked_tokens = row['masked_tokens']

            # Encode masked text for input to the model
            inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = model.generate(inputs["input_ids"], max_new_tokens=128)
            
            predicted_tokens = tokenizer.decode(outputs[0], skip_special_tokens=True)  

            refs.append(masked_tokens)
            preds.append(predicted_tokens)
            
            # Store predictions
            predictions.append({
                "mask_text": masked_text,
                "mask_tokens": masked_tokens,
                "predicted_tokens": predicted_tokens,
            })

        # Calculate BERTScore
        P, R, F1 = score(preds, refs, lang="en", verbose=True)  

        avg_f1_score = F1.mean().item()  
        print(f"Average BERT F1 Score: {avg_f1_score:.2f}")
        return avg_f1_score, predictions

    # Evaluate Before Pre-training
    print("Evaluating model before pre-training...")
    baseline_score, baseline_predictions = evaluate_model(model, test_dataset)

    # Train the Model (Pre-training)
    print("Pre-training model...")
    trainer.train()

    model.save_pretrained("model-codet5/")
    tokenizer.save_pretrained("model-codet5/")

    # Evaluate After Pre-training
    print("Evaluating model after pre-training...")
    _, post_training_predictions = evaluate_model(model, test_dataset)

    # Combine results into a DataFrame
    results_df = pd.DataFrame({
        "mask_text_before": [pred["mask_text"] for pred in baseline_predictions],
        "mask_tokens_before": [pred["mask_tokens"] for pred in baseline_predictions],
        "predicted_tokens_before": [pred["predicted_tokens"] for pred in baseline_predictions],
        "mask_text_after": [pred["mask_text"] for pred in post_training_predictions],
        "mask_tokens_after": [pred["mask_tokens"] for pred in post_training_predictions],
        "predicted_tokens_after": [pred["predicted_tokens"] for pred in post_training_predictions],
    })

    path = args.out
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"Results saved to {path}")

model_name = "Salesforce/codet5-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Code-T5 Model on Code Search Net Dataset")

    parser.add_argument('-e', '--epoch', type=int, default=1, help="Pretrain Parameter: Epoch")
    parser.add_argument('-m', '--mask', type=float, default=.15, help="Pretrain Parameter: Mask Token Portion")
    parser.add_argument('-o', '--out', type=str, default="dataset_v2/generated/evaluation_results.csv", help="Pretrain Parameter: Test Result Path")

    args = parser.parse_args()

    dataset = get_dataset()
    dataset = filter_dataset(dataset)
    dataset = mask_dataset(dataset, mask_rate=args.mask)

    pretrain_model(dataset, args)

    wandb.finish()
