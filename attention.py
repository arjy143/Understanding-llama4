import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

hidden_size = 128 #size of embedding
num_attention_heads = 16
num_key_value_heads = 4 #for efficiency: share key value across 4 heads, but we still have 16 unique Q vectors.
                        #called GQA

head_dim = hidden_size//num_attention_heads #dimension of each attention head
max_position_embeddings = 256 #max sequence expected. big models can now have millions
rope_theta = 10000.0 #for RoPE frequency calc
res_norm_eps = 1e-5 #for RMSnorm

