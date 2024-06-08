import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterSampler
import numpy as np
import matplotlib.pyplot as plt

# Load preprocessed data
train_embeddings = torch.load('./data/train_embeddings.pt')
val_embeddings = torch.load('./data/val_embeddings.pt')
train_ids = torch.load('./data/train_ids.pt')
val_ids = torch.load('./data/val_ids.pt')
train_metadata = torch.load('./data/train_metadata.pt')
val_metadata = torch.load('./data/val_metadata.pt')

class ArxivDataset(Dataset):
    def __init__(self, embeddings, ids, metadata):
        self.embeddings = embeddings
        self.ids = ids
        self.metadata = metadata

    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx], 'id': self.ids[idx], 'metadata': self.metadata[idx]}

    def __len__(self):
        return len(self.ids)

train_dataset = ArxivDataset(train_embeddings, train_ids, train_metadata)
val_dataset = ArxivDataset(val_embeddings, val_ids, val_metadata)

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

# Hyperparameter space
param_grid = {
    'lr': [0.001, 0.0001],
    'hidden_dim': [64, 128, 256],
    'batch_size': [8, 16, 32],
    'num_epochs': [10, 20, 30]
}

param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=0))

best_loss = float('inf')
best_params = None
best_model = None

training_losses = []
validation_losses = []

for params in param_list:
    batch_size = params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = train_embeddings.shape[1]
    hidden_dim = params['hidden_dim']
    model = Autoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            embeddings = batch['embedding']
            _, reconstructed = model(embeddings)
            loss = criterion(reconstructed, embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        training_losses.append(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding']
                _, reconstructed = model(embeddings)
                loss = criterion(reconstructed, embeddings)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        validation_losses.append(epoch_val_loss)
        
    print(f'Params: {params}, Validation Loss: {epoch_val_loss}')
    
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        best_params = params
        best_model = model

print(f'Best Params: {best_params}, Best Validation Loss: {best_loss}')

# Save the best model
torch.save(best_model.state_dict(), './model/best_autoencoder_model.pth')

# Optionally save the entire model
torch.save(best_model, './model/best_autoencoder_entire_model.pth')

# Refine the embeddings using the best autoencoder
best_model.eval()

# Refine train embeddings
refined_train_embeddings = []
with torch.no_grad():
    for batch in DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False):
        embeddings = batch['embedding']
        encoded, _ = best_model(embeddings)
        refined_train_embeddings.extend(encoded.numpy())

refined_train_embeddings = torch.tensor(refined_train_embeddings)
torch.save(refined_train_embeddings, './data/best_refined_train_embeddings.pt')

# Refine val embeddings
refined_val_embeddings = []
with torch.no_grad():
    for batch in DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False):
        embeddings = batch['embedding']
        encoded, _ = best_model(embeddings)
        refined_val_embeddings.extend(encoded.numpy())

refined_val_embeddings = torch.tensor(refined_val_embeddings)
torch.save(refined_val_embeddings, './data/best_refined_val_embeddings.pt')

print("Hyperparameter tuning and model saving complete")

# Plot training and validation losses
epochs = range(len(training_losses))
plt.plot(epochs, training_losses, label='Training Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
