import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], return_tensors="pt", padding=True, truncation=True, max_length=512)

def load_data(file_path):
    try:
        with open(file_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'].squeeze(0) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'].squeeze(0) for item in batch], batch_first=True, padding_value=0)
    return {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}

def embed_batch(batch):
    with torch.no_grad():
        outputs = model(**batch)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def add_embeddings(df, batch_size=32):
    texts = df['abstract'].tolist()
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    embeddings = []
    for batch in tqdm(dataloader, desc="Embedding Text"):
        embeddings.append(embed_batch(batch))

    embeddings_tensor = torch.cat(embeddings).cpu()

    df['embedding'] = embeddings_tensor.numpy().tolist()
    return df

def clean_metadata(metadata):
    def convert_value(value):
        if isinstance(value, (str, int, float, bool)):
            return value
        elif value is None:
            return ""
        elif isinstance(value, list) or isinstance(value, dict):
            return json.dumps(value)  # Convert lists and dicts to JSON strings
        else:
            return str(value)

    return {key: convert_value(value) for key, value in metadata.items()}

def save_to_csv(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")
        raise

if __name__ == "__main__":
    try:
        df = load_data('./data/raw/arxiv-metadata-oai-snapshot.json')
        df = add_embeddings(df)
        save_to_csv(df, './data/processed/arxiv_data.csv')
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
