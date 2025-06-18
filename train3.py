import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding_layer import BasicEmbeddingLayer
from model.attention import SimplifiedLlama4Attention
from model.feedforward import SimplifiedLlama4FFN
from tokeniser.tokeniser import Tokeniser
from model.transformer_block import TransformerBlock

sequence_length = 14
batch_size = 2
tokeniser = Tokeniser(merges=15)
embedding_dim = hidden_size = 128
ffn_intermediate_ratio = 8 / 3
multiple_of = 32
intermediate_size = ((int(hidden_size * ffn_intermediate_ratio) + multiple_of - 1) // multiple_of) * multiple_of

common_config = {
    'hidden_size': hidden_size,
    'num_attention_heads': 16,
    'num_key_value_heads': 4,
    'max_position_embeddings': 256,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'use_qk_norm': True,
    'intermediate_size': intermediate_size,
    'hidden_act': 'silu',
    'ffn_bias': False,
    'rms_norm_eps': 1e-5,
}
config = common_config
vocab_size = len(tokeniser.token_to_id)

model = TransformerBlock(config, vocab_size)

# Prepare inputs
input_ids = torch.tensor(tokeniser.encode("is this document the third one?")).unsqueeze(0).repeat(batch_size, 1)
logits = model(input_ids)

# Get predictions
probs = F.softmax(logits, dim=-1)
predicted_tokens = torch.argmax(probs, dim=-1)

for i in predicted_tokens:
    print(tokeniser.decode(i.tolist()))
