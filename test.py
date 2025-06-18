import torch
import torch.nn.functional as F

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
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


# Example usage:
prompt = "The quick brown fox"
generated = generate_tokens(your_model, your_tokenizer, prompt)
print("Generated text:", generated)
