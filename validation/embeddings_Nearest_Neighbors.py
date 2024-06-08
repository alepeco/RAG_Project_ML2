import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load embeddings
original_embeddings = torch.load('./data/train_embeddings.pt').numpy()
refined_embeddings = torch.load('./data/best_refined_train_embeddings.pt').numpy()

# Find nearest neighbors
num_neighbors = 5
nbrs_orig = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(original_embeddings)
nbrs_refined = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(refined_embeddings)

distances_orig, indices_orig = nbrs_orig.kneighbors(original_embeddings)
distances_refined, indices_refined = nbrs_refined.kneighbors(refined_embeddings)

# Consistency check
consistency = np.mean([np.intersect1d(indices_orig[i], indices_refined[i]).size for i in range(len(indices_orig))]) / num_neighbors
print(f"Nearest Neighbors Consistency: {consistency * 100:.2f}%")
