import torch
import torch.nn as nn


class Attention_layer(nn.Module):
    """Couche de self-attention de base"""

    def __init__(self, d_hidden: int = 1024, causal: bool = True):
        super().__init__()
        self.d_hidden = d_hidden
        self.causal = causal
        self.scale = d_hidden ** -0.5
        self.bound = 1.0 / (d_hidden**0.5)

        self.W_Q = nn.Parameter(torch.empty(d_hidden, d_hidden))
        self.W_K = nn.Parameter(torch.empty(d_hidden, d_hidden))
        self.W_V = nn.Parameter(torch.empty(d_hidden, d_hidden))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.W_Q, -self.bound, self.bound)
        nn.init.uniform_(self.W_K, -self.bound, self.bound)
        nn.init.uniform_(self.W_V, -self.bound, self.bound)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, d_hidden = X.shape
        
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # scores shape: (seq_len, seq_len, batch_size)
        scores = torch.einsum("sbd,tbd->stb", Q, K) * self.scale

        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attention = torch.softmax(scores, dim=1)
        output = torch.einsum("stb,tbd->sbd", attention, V)
        return output


def main():
    layer=Attention_layer(d_hidden = 32, causal = True)

    x= torch.randn(8,1,32)
    y= layer(x)
    print(y)

if __name__ == "__main__":
    main()