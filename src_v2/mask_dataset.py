import torch

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