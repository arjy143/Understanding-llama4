import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

#apply NN directly to each token position after attention mechanism and residual connection.
#up to now, we have just been adding context in the attention mechanism. NN introduces non linearity.

#llms now typically do a layer normalisation step. used to be done after FFNN step.

#config:
hidden_size = 128
#common pattern is around 2.67 * hidden_size, rounded up to a multiple such as 256.
ffn_intermediate_ratio = 8 / 3
multiple_of = 32
intermediate_size = int(hidden_size * ffn_intermediate_ratio)

#adjust intermediate size to be multiple of 32
intermediate_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of

hidden_act = "silu" #activation
rms_norm_eps = 1e-5
ffn_bias = False #use bias in linear layers

#output of Attention + Residual
batch_size = 2
sequence_length = 10
#state before normalisation that we are about to do
input_to_ffn_block = torch.randn(batch_size, sequence_length, hidden_size)

#get rmsnorm
class SimplifiedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) #weight is learnable
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

#the whole thing is in this class
class SimplifiedLlama4FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.hidden_act = config['hidden_act']
        self.ffn_bias = config['ffn_bias']
        self.rms_norm_eps = config['rms_norm_eps']

        #normalisation that we will use
        self.norm = SimplifiedRMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        #mlp layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.ffn_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.ffn_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.ffn_bias)

        if self.hidden_act == "silu":
            self.activation_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation {self.hidden_act} not implemented.")

    def forward(self, hidden_states):
        #apply normalisation
        normalized_states = self.norm(hidden_states)
        #mlp
        gate = self.gate_proj(normalized_states)
        up = self.up_proj(normalized_states)
        down = self.down_proj(self.activation_fn(gate) * up)

        return down

ffn_config_dict = {
    'hidden_size': hidden_size,
    'intermediate_size': intermediate_size,
    'hidden_act': hidden_act,
    'ffn_bias': ffn_bias,
    'rms_norm_eps': rms_norm_eps,
}

simplified_ffn_module = SimplifiedLlama4FFN(ffn_config_dict)

#run forward pass
mlp_output_from_module = simplified_ffn_module(input_to_ffn_block)
#apply residual - feed output back into input
final_output_from_module = input_to_ffn_block + mlp_output_from_module

print("\nOutput shape from simplified FFN module (before residual):", mlp_output_from_module.shape)
print("Output shape after external residual connection:", final_output_from_module.shape)
