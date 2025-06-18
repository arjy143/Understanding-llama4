import torch
import torch.nn.functional as F
from tokeniser.tokeniser import Tokeniser
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
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
model.load_state_dict(torch.load("llm_checkpoint.pt"))
# Optimizer and Loss
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

model.train()

# Prepare inputs (batch_size x seq_len)
input_ids = torch.tensor(tokeniser.encode("is this document the third one?")).unsqueeze(0).repeat(batch_size, 1)

# Targets: next token prediction (shift inputs by one to the left)
# For example, input:  [x1, x2, x3, ..., xN]
# targets:           [x2, x3, ..., xN, <pad or ignore>]
targets = input_ids[:, 1:].clone()
# Pad the last token to avoid size mismatch - fill with -100 so ignored by loss
pad_token_id = -100
targets = torch.cat([targets, torch.full((batch_size, 1), pad_token_id)], dim=1)

# Create causal attention mask (optional, but recommended for autoregressive models)
# seq_len = input_ids.size(1)
# attention_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
# attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

# Forward pass
logits = model(input_ids)

# Reshape logits and targets for loss
# logits: (batch_size, seq_len, vocab_size) -> (batch_size*seq_len, vocab_size)
logits_flat = logits.view(-1, vocab_size)
targets_flat = targets.view(-1)

loss = criterion(logits_flat, targets_flat)

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
torch.save(model.state_dict(), "llm_checkpoint.pt")