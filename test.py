import torch
import torch.nn.functional as F
from tokeniser.tokeniser import Tokeniser
from model.transformer_block import TransformerBlock

def generate_tokens(model, tokenizer, prompt_text, max_length=50, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  #(1, seq_len)
    generated_ids = input_ids.tolist()[0] 

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)  # (1, seq_len, vocab_size)
        logits = outputs[:, -1, :]  # Take the logits for the last token: shape (1, vocab_size)
        # softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample from the distribution or take argmax (greedy)
        next_token_id = torch.argmax(probs, dim=-1).item()
        # Alternatively, use sampling:
        # next_token_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token_id)
        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        if next_token_id == tokenizer.eod:
            break

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

sequence_length = 16
batch_size = 2
tokeniser = Tokeniser(merges=15)

import json
with open('data\common_config.json', 'r') as f:
    config = json.load(f)

vocab_size = len(tokeniser.token_to_id)
model = TransformerBlock(config, vocab_size)
model.load_state_dict(torch.load("llm_checkpoint.pt"))
tokeniser = Tokeniser(merges=10000)

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:10]
data = dataset["text"]

# the below prompt actually generates some readable text somehow
prompt = "which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in"
#prompt = "is this document the third one?"
generated = generate_tokens(model, tokeniser, prompt, max_length=50)
print("Generated text:", generated)
