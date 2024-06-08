import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('./data/processed/arxiv_data.csv')

# Ensure IDs are strings
df['id'] = df['id'].astype(str)

# Replace NaN values with empty strings
df = df.fillna('')

# Convert embeddings from string to list
df['embedding'] = df['embedding'].apply(eval)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract embeddings and IDs
train_embeddings = torch.tensor(train_df['embedding'].tolist())
val_embeddings = torch.tensor(val_df['embedding'].tolist())
train_ids = train_df['id'].tolist()
val_ids = val_df['id'].tolist()

# Create metadata including abstract
train_metadata = train_df.drop(columns=['embedding']).to_dict(orient='records')
val_metadata = val_df.drop(columns=['embedding']).to_dict(orient='records')

# Save the preprocessed data
torch.save(train_embeddings, './data/train_embeddings.pt')
torch.save(val_embeddings, './data/val_embeddings.pt')
torch.save(train_ids, './data/train_ids.pt')
torch.save(val_ids, './data/val_ids.pt')
torch.save(train_metadata, './data/train_metadata.pt')
torch.save(val_metadata, './data/val_metadata.pt')
