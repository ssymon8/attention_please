import torch
import torch.nn as nn

"""La MQA c'est de la MHA sauf que les 'têtes d'attention' 
sont les Key et Value se partagent sur toutes les têtes de Query MQA(Q,K,V) = Concat_h(Att(Q_h, K, V))

On a Q = (d_hidden/n_heads, d_hidden/n_heads, n_heads)
     K = (d_hidden/n_heads, d_hidden/n_heads)
     V = (d_hidden/n_heads, d_hidden/n_heads)
"""

class MQA_layer(nn.Module):

    def __init__(self, d_hidden : int = 1024, n_heads: int = 32, causal : bool = True):
        super().__init__()

        assert d_hidden%n_heads == 0

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_heads = d_hidden// n_heads
        self.causal = causal
        self.scale = d_hidden ** -0.5
        self.bound = 1.0 / (self.d_heads**0.5)

        self.W_Q = nn.ModuleList(nn.Parameter(torch.empty(d_hidden, self.d_heads)) for _ in range(n_heads))
        self.W_K = nn.Parameter(torch.empty(d_hidden, self.d_heads))
        self.W_V = nn.Parameter(torch.empty(d_hidden, self.d_heads))


        self.reset_parameters()

    def reset_parameters(self) -> None:
        for wq in self.W_Q:
            nn.init.uniform_(wq, -self.bound, self.bound)
        nn.init.uniform_(self.W_K, -self.bound, self.bound)
        nn.init.uniform_(self.W_V, -self.bound, self.bound)

    def forward(self, X):
        Qs = [X @ wq for wq in self.W_Q]
        K = X @ self.W_K
        V = X @ self.W_V

        outputs = []

        for Q in Qs:
            scores = torch.einsum("sbd,tbd->stb", Q, K) * self.scale

            if self.causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
            
            attn = torch.softmax(scores, dim=1)
            out = torch.einsum("stb,tbd->sbd", attn, V)
            outputs.append(out)
        
        return torch.cat(outputs, dim = -1)

