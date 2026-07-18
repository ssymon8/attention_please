import torch
import torch.nn.functional as F
from language_model_from_scratch import language_model
from tokenizers import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_layers = 3
d_hidden= 1024
n_heads = 32
n_groups = 8

tokenizer = Tokenizer.from_file("hugo_tokenizer.json")

model = language_model(num_layers = num_layers, n_heads = n_heads, n_groups = n_groups, d_hidden = d_hidden, vocab_size = tokenizer.get_vocab_size()).to(device)

checkpoint_path = "./checkpoints2/checkpoint_epoch50.pt" 
checkpoint = torch.load(checkpoint_path, map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

def generate_text(model, prompt, tokenizer, max_tokens: int = 128, temperature : float = 1.0, top_k: int = 50, context_len = 521) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype = torch.long, device = device)

    with torch.no_grad():
        for _ in range(max_tokens):
            #sliding context window
            context_ids = input_ids[:, -context_len:]

            logits = model(context_ids)

            next_token_logits = logits[:,-1,:]

            next_token_logits = next_token_logits / max(temperature, 1e-5)

            #top-k filter
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            probas = F.softmax(next_token_logits, dim= 1)

            next_token = torch.multinomial(probas, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == 3: #id of EOS 
                break
    
    return tokenizer.decode(input_ids[0].tolist())


if __name__ == '__main__':

    prompt="J'aime voir le ciel bleu, "

    generated_text = generate_text(model, prompt, tokenizer, temperature = 0.75, max_tokens = 64)

    print(generated_text)

                

