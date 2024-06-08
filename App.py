import torch
from flask import Flask, request, jsonify, render_template
from src.embedding_utils import query_to_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load refined embeddings and metadata
refined_train_embeddings = torch.load('./data/best_refined_train_embeddings.pt')
train_ids = torch.load('./data/train_ids.pt')
train_metadata = torch.load('./data/train_metadata.pt')

# Convert embeddings to numpy for similarity calculations
refined_train_embeddings_np = refined_train_embeddings.numpy()

@app.route('/')
def home():
    return render_template('index.html')

def find_similar_embeddings(query_embedding, top_k=5):
    query_embedding_np = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_np, refined_train_embeddings_np)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    return top_indices

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data['query']
    
    try:
        # Generate the embedding from the query
        embedding = query_to_embedding(query_text)
        
        # Find the most similar embeddings
        top_indices = find_similar_embeddings(embedding)
        
        # Prepare the response
        response_data = []
        for idx in top_indices:
            metadata = train_metadata[idx].copy()
            abstract = metadata.pop('abstract', 'No abstract available')
            response_data.append({
                'id': train_ids[idx],
                'metadata': metadata,
                'abstract': abstract
            })
        
        # Debug: Log the response data
        print(f"Response data: {response_data}")
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
