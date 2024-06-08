import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load preprocessed data
train_embeddings = torch.load('./data/train_embeddings.pt')
val_embeddings = torch.load('./data/val_embeddings.pt')
refined_train_embeddings = torch.load('./data/best_refined_train_embeddings.pt')
refined_val_embeddings = torch.load('./data/best_refined_val_embeddings.pt')

# Concatenate train and validation embeddings
original_embeddings = torch.cat((train_embeddings, val_embeddings)).numpy()
refined_embeddings = torch.cat((refined_train_embeddings, refined_val_embeddings)).numpy()

# Compute cosine similarities
original_similarities = cosine_similarity(original_embeddings)
refined_similarities = cosine_similarity(refined_embeddings)

# Print the average cosine similarities
print(f"Average Original Cosine Similarity: {original_similarities.mean()}")
print(f"Average Refined Cosine Similarity: {refined_similarities.mean()}")

# Optionally, you can also visualize the similarities or other statistics
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Plot histograms of cosine similarities
plt.subplot(1, 2, 1)
plt.hist(original_similarities.flatten(), bins=50, alpha=0.7, label='Original')
plt.title('Original Embeddings Cosine Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(refined_similarities.flatten(), bins=50, alpha=0.7, label='Refined')
plt.title('Refined Embeddings Cosine Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
