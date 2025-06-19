import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tokeniser.tokeniser import Tokeniser
from model.transformer_block import TransformerBlock

# ------------------ Config ------------------
sequence_length = 16
batch_size = 2
num_epochs = 10
tokeniser = Tokeniser(merges=15)

import json
with open('data\common_config.json', 'r') as f:
    config = json.load(f)

vocab_size = len(tokeniser.token_to_id)

# ------------------ Model ------------------
model = TransformerBlock(config, vocab_size)
###use the below code to load the current model weights if you dont want to train from scratch
# try:
#     model.load_state_dict(torch.load("llm_checkpoint.pt"))
#     print("Loaded checkpoint.")
# except FileNotFoundError:
#     print("Training from scratch.")

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss(ignore_index=-100)
model.train()

# ------------------ Data Preparation ------------------
# data = [
#     "this is the first example",
#     "here is another sentence",
#     "the model will learn from this",
#     "more and more data helps it improve",
#     "adding more lines to simulate a dataset",
# ]
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")[:10]
data = dataset["text"]
# Tokenize all data
all_tokens = []
for line in data:
    all_tokens.extend(tokeniser.encode(line))

# Chunk into sequences of length (sequence_length + 1)
def chunk_tokens(token_ids, seq_len):
    return [token_ids[i:i+seq_len] for i in range(0, len(token_ids) - seq_len)]

class TokenDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.sequences = chunk_tokens(token_ids, seq_len + 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        targets = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, targets

dataset = TokenDataset(all_tokens, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------ Training Loop ------------------
for epoch in range(num_epochs):
    total_loss = 0
    for input_ids, targets in dataloader:
        logits = model(input_ids)  # (batch, seq_len, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "llm_checkpoint.pt")
