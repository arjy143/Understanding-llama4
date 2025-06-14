import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

#GQA = grouped query attention = fewer K and V than Q 

hidden_size = 128 #size of embedding
num_attention_heads = 16
num_key_value_heads = 4 #for efficiency: share key value across 4 heads, but we still have 16 unique Q vectors.
                        #called GQA

head_dim = hidden_size//num_attention_heads #dimension of each attention head
max_position_embeddings = 256 #max sequence expected. big models can now have millions
rope_theta = 10000.0 #for RoPE frequency calc
res_norm_eps = 1e-5 #for RMSnorm

attention_bias = False #no bias used in attention
attention_dropout = 0 #no need - prevents overfitting
use_qk_norm = True #whether to apply L2 norm to Q and K before attention

batch_size = 2 #2 independent sequences of text
sequence_length = 10 #context window of tokens - amount of tokens you look back at
hidden_states = torch.randn(batch_size, sequence_length, hidden_size) #10 hidden states
#below, we create a 1D tensor [0...sequence_length-1]
#then we unsqueeze to add a extra dimension at the 0th position, making it (1, sequence_length)
#then we repeat(2,1) which creates tensor of shape (2, 10)
position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1)

#mask = upper triangular matrix. in reality, is more complicated.
#we put -inf in the relevant positions
attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) #shape (1,1,10,10)
attention_mask = attention_mask.expand(batch_size, 1, -1, -1) #shape (2,1,10,10)

#below are the weight matrices, aka projection layers
q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
#o_proj is output projection - used after head concatenation and before residual stuff, if we're doing that

#calculate Q, K, V
query_states = q_proj(hidden_states)
key_states = k_proj(hidden_states)
value_states = v_proj(hidden_states)

#reshape Q, K, V for multi head attention
#will end up with shape (batch_size, num_heads, sequence_length, head_dim)
query_states = query_states.view(batch_size, sequence_length, num_attention_heads, head_dim).transpose(1,2)
key_states = key_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1,2)
value_states = value_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1,2)

num_key_value_groups = num_attention_heads // num_key_value_heads

###rotary positional embeddings
#use rotations to embed positional information before the Q and K dot product

#represent embeddings in complex number space and rotate them to represent their postions
#dim: must be even because rope works on pairs
#base: controls frequency scaling
def simple_rope_calculation(dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))#creates indices starting at 0, going up by 2, and eventually getting reciprocal
    t = torch.arange(max_seq_len).type_as(inv_freq) 
    freqs = torch.outer(t, inv_freq) #compute outer product. produces matrix of angles for each position and frequency
        
    #the following is for the rotation calculation
    emb = torch.cat((freqs, freqs), dim=1)
    freqs_cos = emb.cos()
    freqs_sin = emb.sin()
    #complex number represents rotation by angle in complex space
    freqs_cis = torch.complex(freqs_cos, freqs_sin)
    #return the tensor containing all rotations for positions up to max_seq_len and dim
    return freqs_cis

def new_func(inv_freq, t):
    freqs = torch.outer(t, inv_freq)
    return freqs

#apply rotations to Q and K
def apply_rotary_emb_torch(xq, #query
                           xk, #key
                           freqs_cis #rotations
                           ):
    #only select rotation values for the actual token positions
    freqs_cis = freqs_cis[position_ids]
    freqs_cis = freqs_cis[:, None, :, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broadcast = freqs_cis[..., :xq_.shape[-1]]

    rotated_xq = xq_ * freqs_cis_broadcast
    rotated_xk = xk_ * freqs_cis_broadcast
    xq_out = torch.view_as_real(rotated_xq).flatten(3)
    xk_out = torch.view_as_real(rotated_xk).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

#calculate rope
freqs_cis = simple_rope_calculation(head_dim, max_position_embeddings, base=rope_theta)
print(f"freqs_cis shape {freqs_cis.shape}")

#apply rope
query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, freqs_cis)
print(f"Q shape {query_states_rope.shape}")
print(f"K shape {key_states_rope.shape}")

