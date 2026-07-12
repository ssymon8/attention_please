from transformer_layer import transformer

import torch
import torch.nn as nn
import torch.nn.functional as F

class language_model(nn.Module):
    def __init__(self, num_layers, d_hidden, n_groups, n_heads, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_hidden)

        self.blocks = nn.ModuleList([transformer(d_hidden = d_hidden, n_heads = n_heads, n_groups = n_groups) for _ in range(num_layers)])

        self.lm_head = nn.Linear(d_hidden, vocab_size, bias = False)

    def forward(self, X):
        B, T = X.shape

        h = self.token_embedding(X)

        for block in self.blocks:
            h = block(h)
        
        logits = self.lm_head(h)
        return logits

if __name__ == "__main__":

    vocab_size = 10
    
    X = torch.Tensor([[1,2,5,6,2,4]]).long() #B= 1, T = 6

    lm = language_model(num_layers = 2, d_hidden = 256, n_groups = 2, n_heads = 4, vocab_size = vocab_size)

    print(f"input tensor {X} outputs {lm(X)}")

