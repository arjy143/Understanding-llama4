import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding_layer import BasicEmbeddingLayer
from model.attention import SimplifiedLlama4Attention
from model.feedforward import SimplifiedLlama4FFN
from tokeniser.tokeniser import Tokeniser

class TransformerBlock(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.hidden_size = config['hidden_size']
        
        self.embedding = BasicEmbeddingLayer(vocab_size, self.hidden_size)
        self.attention = SimplifiedLlama4Attention(config)
        self.ffn = SimplifiedLlama4FFN(config)
        self.lm_head = nn.Linear(self.hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # Embedding tokens + positions
        hidden_states = self.embedding(input_ids)
        
        # Default position_ids if None
        if position_ids is None:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        
        # Default attention mask: causal mask if not provided
        if attention_mask is None:
            seq_len = input_ids.size(1)
            # Upper triangular mask with -inf on future positions, 0 on allowed
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(input_ids.size(0), 1, seq_len, seq_len)

        # Attention + Residual
        attn_output, _ = self.attention(hidden_states, attention_mask, position_ids)
        attn_output = attn_output + hidden_states

        # Feed Forward + Residual
        ffn_output = self.ffn(attn_output)
        final_output = ffn_output + attn_output

        # LM head to produce logits
        logits = self.lm_head(final_output)
        
        return logits