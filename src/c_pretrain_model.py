import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import random
from sklearn.model_selection import train_test_split
from bert_score import score

MASK_RATE = 0.15
model_name = "Salesforce/codet5-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

data_path = "dataset/generated/flatten_dataset.csv"
df = pd.read_csv(data_path)
texts = df['cleaned_method'].tolist()
texts = [t.strip('\"') for t in texts if type(t) == str and len(tokenizer.tokenize(t)) < 500]

texts, test_texts = train_test_split(texts, test_size=0.01, random_state=42)
test_df = pd.DataFrame({"text": test_texts}).sample(min(100, len(test_texts)), random_state=42)

def mask_text(text, mask_rate=MASK_RATE, seed=42):
    random.seed(seed) 
    tokens = tokenizer.tokenize(text) 
    num_to_mask = max(1, int(len(tokens) * mask_rate))
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    for i in mask_indices:
        tokens[i] = "<extra_id_0>"
    return " ".join(tokens)

test_df['mask_text'] = test_df['text'].apply(mask_text)

print(f"\n\nTrain Method Count: {len(texts)}\n")

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, max_length=128):
        self.texts = texts
        self.max_length = max_length  
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        masked_text = mask_text(text)
        inputs = tokenizer(
            masked_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length
        )
        labels = tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length
        )
        return {
            "input_ids": inputs.input_ids[0],
            "labels": labels.input_ids[0]
        }

dataset = MaskedDataset(texts)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="models/pretrained_codet5",
    num_train_epochs=3,
    logging_steps=10,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer for Pre-training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

def evaluate_model(model, test_df, sample_size=100):
    model.eval()
    predictions = []

    refs = []
    preds = []

    # Iterate through rows in test_df
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        text = row['text']  
        masked_text = row['mask_text']  

        # Encode masked text for input to the model
        inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_new_tokens=1000)
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  

        refs.append(text)
        preds.append(predicted_text)       
        
        # Store predictions
        predictions.append({
            "text": text,
            "mask_text": masked_text,
            "predicted_text": predicted_text,
        })

    # Calculate BERTScore
    P, R, F1 = score(preds, refs, lang="en", verbose=True)  

    avg_f1_score = F1.mean().item()  
    print(f"Average BERT F1 Score: {avg_f1_score:.2f}")
    return avg_f1_score, predictions

# 1. Evaluate Before Pre-training
print("Evaluating model before pre-training...")
baseline_score, baseline_predictions = evaluate_model(model, test_df)

# 2. Train the Model (Pre-training)
print("Pre-training model...")
trainer.train()

model.save_pretrained("models/pretrained_codet5/final_model")
tokenizer.save_pretrained("models/pretrained_codet5/final_model")

# 3. Evaluate After Pre-training
print("Evaluating model after pre-training...")
post_training_score, post_training_predictions = evaluate_model(model, test_df)

# Combine results into a DataFrame
results_df = pd.DataFrame({
    "text": [pred["text"] for pred in baseline_predictions],
    "mask_text_before": [pred["mask_text"] for pred in baseline_predictions],
    "predicted_text_before": [pred["predicted_text"] for pred in baseline_predictions],
    "mask_text_after": [pred["mask_text"] for pred in post_training_predictions],
    "predicted_text_after": [pred["predicted_text"] for pred in post_training_predictions],
})

# Save the results to CSV
results_df.to_csv("dataset/generated/evaluation_results.csv", index=False)
print("Results saved to dataset/generated/evaluation_results.csv")
