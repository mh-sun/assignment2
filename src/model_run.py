import os
import pandas as pd
import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Initialize W&B
wandb.init(project="code-generation-fill-in", name="finetune-codellama")

def data_split(src_dataset, train_set, val_set, test_set, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    data = pd.read_csv(src_dataset)

    try:
        data = data.dropna()

        sample_size = min(50000, len(data))
        data = data.sample(sample_size, random_state=42)
    except:
        print(f"\n\n\n### Not enough data for fine-tuning. ###\n\n\n")
        return

    train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=42, shuffle=True)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42, shuffle=True)

    train_length = len(train_data)
    val_length = len(val_data)
    test_length = len(test_data)

    # Save the DataFrames to CSV files
    train_data.to_csv(train_set, index=False)
    val_data.sample(500).to_csv(val_set, index=False)
    test_data.sample(500).to_csv(test_set, index=False)

    # Print lengths and confirmation messages
    print(f"Data split completed:\n- Training set saved to {train_set} (Length: {train_length})\n- Validation set saved to {val_set} (Length: {val_length})\n- Test set saved to {test_set} (Length: {test_length})")

def fine_tune_model(train_file, val_file, test_file, model_name="codellama/CodeLlama-7b-hf", output_dir="./fine_tuned_codellama"):
    dataset = load_dataset("csv", data_files={"train": train_file, "validation": val_file, "test": test_file})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    def tokenize_function(examples):
        inputs = examples["masked_method"]
        targets = examples["condition_line"]

        # Tokenize inputs and targets with truncation and padding
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128, return_tensors="pt")["input_ids"]

        # Assign labels to model_inputs
        model_inputs["labels"] = labels

        return model_inputs
    
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_datasets = dataset.map(tokenize_function, batch_size=32, batched=True, remove_columns=["masked_method", "condition_line"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        logging_strategy="steps",
        max_steps=30,
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="wandb",  # Enable W&B logging
        save_total_limit=2,
        save_steps=500,
    )
    
    # Define trainer with W&B integration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model on test set and log results in W&B
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test set evaluation:", test_results)
    wandb.log(test_results)
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")
    wandb.finish()

if __name__ == "__main__":
    # Define file paths
    src_dataset = "dataset/flatten_dataset.csv"
    train_set = "dataset/train_dataset.csv"
    val_set = "dataset/val_dataset.csv"
    test_set = "dataset/test_dataset.csv"
    
    # Split data
    data_split(src_dataset, train_set, val_set, test_set)
    
    # Fine-tune model and log in W&B
    fine_tune_model(train_file=train_set, val_file=val_set, test_file=test_set)
