# pinecone_setup.py

import os
from dotenv import load_dotenv
import pandas as pd
from langchain.vectorstores import Pinecone as LangchainPinecone

# Load environment variables from the .env file
load_dotenv()

# Access the API key from the environment variable
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=api_key)

# Create Pinecone index
index_name = "full-arxiv-embeddings"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Assuming embedding dimension is 768
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Load your data
df = pd.read_csv('./data/processed/arxiv_data.csv')

# Ensure IDs are strings
df['id'] = df['id'].astype(str)

# Replace NaN values with empty strings and remove newlines
df = df.fillna('')
df['abstract'] = df['abstract'].str.replace('\n', ' ').str.strip()

# Prepare data for insertion
ids = df['id'].tolist()
embeddings = [list(map(float, emb.strip("[]").split(','))) for emb in df['embedding'].tolist()]
metadata = df.drop(columns=['id', 'embedding']).to_dict(orient='records')

# Convert metadata to string to avoid any type issues
metadata = [{k: str(v) for k, v in record.items()} for record in metadata]

# Add data to Pinecone in batches
batch_size = 100  # Adjust the batch size as needed
for i in range(0, len(ids), batch_size):
    batch_vectors = [{'id': ids[j], 'values': embeddings[j], 'metadata': metadata[j]} for j in range(i, min(i + batch_size, len(ids)))]
    index.upsert(vectors=batch_vectors)

# Initialize LangChain's Pinecone vectorstore
vector_store = LangchainPinecone(index=index, embedding=embeddings, text_key='abstract')
