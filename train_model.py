import torch
import torch.optim as optim
import torch.nn as nn
from build_model import TransformerModel
from tokenizers import Tokenizer

# Load tokenizer to get vocab size
tokenizer = Tokenizer.from_file("custom_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

# Load the model structure and update vocab size
model = torch.load('transformer_model_structure.pth')
model.fc_out = nn.Linear(512, VOCAB_SIZE)
model.embedding = nn.Embedding(VOCAB_SIZE, 512)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataloader
dataloader = torch.load('dataloader.pth')

# Training hyperparameters
EPOCHS = 3
LR = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for input_ids, target_ids in dataloader:
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        optimizer.zero_grad()
        output = model(input_ids, input_ids)
        loss = criterion(output.view(-1, VOCAB_SIZE), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}')

# Save the model
torch.save(model.state_dict(), 'transformer_model.pth')
