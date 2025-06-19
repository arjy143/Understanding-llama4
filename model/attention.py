import torch
import torch.nn as nn
import math

class SimplifiedLlama4Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config['max_position_embeddings']
        self.rope_theta = config['rope_theta']
        self.attention_bias = config['attention_bias']
        self.use_qk_norm = config['use_qk_norm']

        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        self.freqs_cis = self._simple_rope_calculation(self.head_dim, self.max_position_embeddings, base=self.rope_theta)

        if self.use_qk_norm:
             self.qk_norm = SimpleL2Norm()

    def forward(self, hidden_states, attention_mask, position_ids):
        batch_size, sequence_length, _ = hidden_states.shape

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        current_freqs_cis = self.freqs_cis.to(hidden_states.device) # Get precomputed freqs
        query_states_rope, key_states_rope = self._apply_rotary_emb_torch(query_states, key_states, current_freqs_cis, position_ids)

        # Optional QK Norm
        if self.use_qk_norm:
             query_states_final = self.qk_norm(query_states_rope)
             key_states_final = self.qk_norm(key_states_rope)
        else:
            query_states_final = query_states_rope
            key_states_final = key_states_rope


        # Repeat K/V for GQA
        key_states_repeated = self._repeat_kv(key_states_final, self.num_key_value_groups)
        value_states_repeated = self._repeat_kv(value_states, self.num_key_value_groups)

        # Attention Calculation
        attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = attn_weights * scaling_factor

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        # Dropout would be here in training

        attn_output = torch.matmul(attn_weights, value_states_repeated)

        # Reshape and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.hidden_size)
        final_attn_output = self.o_proj(attn_output)

        return final_attn_output, attn_weights # Return weights for inspection
    
    def _simple_rope_calculation(self, dim, max_seq_len, base=10000.0):
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

#apply rotations to Q and K
    def _apply_rotary_emb_torch(self,
                                xq, #query
                                xk, #key
                                freqs_cis, #rotations
                                position_ids):
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
    
    #repeat K/V heads for GQA, because there are only 4 unique K/V and 16 unique Q
    def _repeat_kv(self, hidden_states, n_rep):
        batch, num_key_value_heads,slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    #normalise input x along its last dimension.
    #computes euclidean norm and scales by inverse of norm
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)