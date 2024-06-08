import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import numpy as np

# Same Autoencoder class definition as used during training
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_dim = 768 
hidden_dim = 256  # Best performing hidden_dim

# Load the autoencoder model
model_path = './model/best_autoencoder_model.pth'
autoencoder = Autoencoder(input_dim, hidden_dim)
autoencoder.load_state_dict(torch.load(model_path))
autoencoder.eval()

# Initialize the tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

def query_to_embedding(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Take the embedding of the [CLS] token
        encoded, _ = autoencoder(embeddings)
        
    # Convert to numpy array and normalize
    embedding_np = encoded[0].cpu().numpy()
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(embedding_np)):
        raise ValueError("Embedding contains NaN or infinite values")
    
    # Normalize embedding
    norm = np.linalg.norm(embedding_np)
    if norm != 0:
        embedding_np = embedding_np / norm
    
    return embedding_np

# Example usage for testing
# if __name__ == "__main__":
    query = "quantum physics"
    embedding = query_to_embedding(query)
    print(embedding)
    print(f"Embedding dimension: {embedding.shape[0]}, Example values: {embedding[:5]}")
