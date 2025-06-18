import torch
import torch.nn as nn
from model.embedding_layer import BasicEmbeddingLayer
from model.attention import SimplifiedLlama4Attention
from model.feedforward import SimplifiedLlama4FFN
from tokeniser.tokeniser import Tokeniser

sequence_length = 14
batch_size = 2
tokeniser = Tokeniser(merges=15)
vocab_size = len(tokeniser.token_to_id)
token_list = tokeniser.encode("is this document the third one?")
input_ids = torch.tensor(token_list).unsqueeze(0).repeat(batch_size, 1)

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

embedding_layer = BasicEmbeddingLayer(vocab_size, hidden_size)
attention_layer = SimplifiedLlama4Attention(common_config)
ffn_layer = SimplifiedLlama4FFN(common_config)

# 4. Prepare input
hidden_states = embedding_layer(input_ids) 
#position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
position_ids = torch.arange(0, sequence_length, device=hidden_states.device).unsqueeze(0)
attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, sequence_length, sequence_length)

# 5. Transformer Block Forward Pass
# ---- Attention + Residual ----
attn_output, _ = attention_layer(hidden_states, attention_mask, position_ids)
attn_output = attn_output + hidden_states  # Residual

# ---- FFN + Residual ----
ffn_output = ffn_layer(attn_output)
final_output = ffn_output + attn_output  # Residual

print("Final transformer block output shape:", final_output.shape)

lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
logits = lm_head(final_output)

probs = torch.softmax(logits, dim=-1)
predicted_tokens = torch.argmax(probs, dim=-1)
print(predicted_tokens)

for i in predicted_tokens:
    tokeniser.decode(i.tolist())