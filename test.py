import torch
import torch.nn.functional as F
from tokeniser.tokeniser import Tokeniser
from model.transformer_block import TransformerBlock

def generate_tokens(model, tokenizer, prompt_text, max_length=50, device='cpu'):
    model.eval()  # Set model to eval mode
    input_ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, seq_len)

    generated_ids = input_ids.tolist()[0]  # start with prompt tokens

    for _ in range(max_length):
        # Run the model on current tokens
        with torch.no_grad():
            outputs = model(input_ids)  # outputs: (1, seq_len, vocab_size)
        
        logits = outputs[:, -1, :]  # Take the logits for the last token: shape (1, vocab_size)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution or take argmax (greedy)
        next_token_id = torch.argmax(probs, dim=-1).item()
        # Alternatively, use sampling:
        # next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Append predicted token
        generated_ids.append(next_token_id)

        # Prepare input for next iteration
        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)

        # Stop if end of sentence token generated (e.g., tokenizer.eos_token_id)
        if next_token_id == tokenizer.eod:
            break

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


# Example usage:
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
tokeniser = Tokeniser(merges=10000)

# from datasets import load_dataset
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:10]
# data = dataset["text"]
# print(data)
prompt = "what is the capital of France?"
#prompt = "is this document the third one?"
generated = generate_tokens(model, tokeniser, prompt, max_length=50)
print("Generated text:", generated)
