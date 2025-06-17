import torch.nn as nn
    
class BasicEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)