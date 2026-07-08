import torch
import torch.nn as nn

"""
La GQA est un peu la forme finale d'optimisation de l'attention (avant la Flash Attention),
c'est comme la MQA sauf que désormais, on groupe les queries sur un groupe de KEY/VALUE

Ainsi, si on a n_g groupes, on répartis les n_h Q dans les groupes puis on concatène tous les résultats
pour obtenir nos score d'attention finaux.

Les Queries sont assignées à leurs K/V par des matrices de projections
"""

class GQA_layer(nn.Module):
    def __init__(self, d_hidden : int = 1024, n_heads: int = 32, n_groups = 8,causal : bool = True):
        super().__init__()

        assert d_hidden%n_heads == 0
        assert n_heads%n_groups == 0 

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_heads = d_hidden// n_heads
        self.causal = causal
        self.scale = self.d_heads ** -0.5
        self.bound = 1.0 / (self.d_heads**0.5)

        self.W_Q = nn.ParameterList(nn.Parameter(torch.empty(d_hidden, self.d_heads)) for _ in range(n_heads))
        self.W_K = nn.ParameterList(nn.Parameter(torch.empty(d_hidden, self.d_heads)) for _ in range(n_groups))
        self.W_V = nn.ParameterList(nn.Parameter(torch.empty(d_hidden, self.d_heads)) for _ in range(n_groups))

        self.reset_parameters()
    
    def reset_parameters(self) -> None:

        for wq in self.W_Q:
            nn.init.uniform_(wq, -self.bound, self.bound)

        for wk in self.W_K:
            nn.init.uniform_(wk, -self.bound, self.bound)

        for wv in self.W_V:
            nn.init.uniform_(wv, -self.bound, self.bound)
    
    def forward(self, X : torch.Tensor):
        seq_len, batch_size, _ = X.shape

        Qs = [X @ wq for wq in self.W_Q]
        Ks = [X @ wk for wk in self.W_K]
        Vs = [X @ wv for wv in self.W_V]

        outputs=[]

        for head_idx, Q in enumerate(Qs):
            group_idx = head_idx // (self.n_heads//self.n_groups)

            scores = scores = torch.einsum("sbd,tbd->stb", Q, Ks[group_idx]) * self.scale

            if self.causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
            
            attn = torch.softmax(scores, dim=1)
            out = torch.einsum("stb,tbd->sbd", attn, Vs[group_idx])
            outputs.append(out)
              
        return torch.cat(outputs, dim = -1)


if __name__ == "__main__":

    d_hidden = 1024
    n_heads = 32
    n_groups = 8

    X = torch.randn(8,1,d_hidden)

    layer = GQA_layer(d_hidden = d_hidden, n_heads = n_heads, n_groups = n_groups, causal = True)

    output = layer(X)

    print(f"Our input X of shape {X.shape} gives output {output} of dimensions {output.shape}")