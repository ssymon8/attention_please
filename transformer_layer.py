import torch
import torch.nn as nn
from SwiGLU_ffn import SwiGLU

#from grouped_query_attention import GQA_layer

class GQA_layer(nn.Module):
    def __init__(self, d_hidden, n_heads, n_groups, causal : bool = True):
        super().__init__()

        assert d_hidden % n_heads == 0
        assert n_heads % n_groups == 0 

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_heads = d_hidden // n_heads
        self.causal = causal
        self.scale = self.d_heads ** -0.5
        self.queries_per_group = n_heads // n_groups

        self.W_Q = nn.Linear(d_hidden, n_heads * self.d_heads, bias=False)
        self.W_K = nn.Linear(d_hidden, n_groups * self.d_heads, bias=False)
        self.W_V = nn.Linear(d_hidden, n_groups * self.d_heads, bias=False)

        self.W_O = nn.Linear(d_hidden, d_hidden, bias=False)
    
    def forward(self, X: torch.Tensor):
        B, T, _ = X.shape

        q=self.W_Q(X)
        k=self.W_K(X)
        v=self.W_V(X)

        q= q.view(B, T, self.n_groups, self.queries_per_group, self.d_heads)
        k = k.view(B, T, self.n_groups, 1, self.d_heads)
        v = v.view(B, T, self.n_groups, 1, self.d_heads)

        positions = torch.arange(T, device = X.device)
        inv_freq = 1.0/ (10000**(torch.arange(0, self.d_heads, 2, device = X.device).float()/self.d_heads))
        freqs = torch.einsum('i, j -> ij', positions.float(), inv_freq) #on se retrouve avec une matrice des position* theta dim (T, d_heads/2)

        cos = torch.cos(freqs).repeat_interleave(2, dim= -1) #on répète 2 fois toutes les fréquences pour avoir dim = (T, d_heads)
        sin = torch.sin(freqs).repeat_interleave(2, dim= -1)

        #ici on obtient des matrices de positionnal embeddings de la mm taille qu'une matrice q/k d'une head
        #on vva reshape pour broadcast sur nos queries values

        cos = cos.view(1, T, 1, 1, self.d_heads)
        sin = sin.view(1, T, 1, 1, self.d_heads)

        q_rot = self.apply_RoPE(q, cos, sin)
        k_rot = self.apply_RoPE(k, cos, sin)

        scores = torch.einsum('btsgd, bnsgd -> bgstn', q_rot, k_rot) * self.scale

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device= X.device, dtype=torch.bool), diagonal = 1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim= -1)

        out = torch.einsum('bgstn, bnsgd-> btsgd', attn, v)

        out = out.reshape(B, T, self.n_heads * self.d_heads)

        return self.W_O(out)
    
    def rotate_half(self, X: torch.Tensor):
        d= X.shape[-1]
        x0, x1 = X[..., :d//2], X[...,d//2:]
        return torch.cat((-x1, x0), dim = -1)

    def apply_RoPE(self, X: torch.Tensor, cos, sin):
        X_rot = X* cos + self.rotate_half(X) * sin
        return X_rot


class transformer(nn.Module):
    def __init__(self, d_hidden : int = 1024, n_heads : int = 32, n_groups : int = 8):
        super().__init__()
        #attention layer
        self.rms_norm_ln = nn.RMSNorm((d_hidden,), eps = 1e-5)
        self.attention_layer = GQA_layer(d_hidden, n_heads, n_groups, causal = True)
        #RMSnorm
        self.rms_norm_ffn = nn.RMSNorm((d_hidden,), eps = 1e-5)
        #our GeLu MLP
        #self.MLP = nn.Sequential(
        #    nn.Linear(d_hidden, 4*d_hidden),
        #    nn.GELU(),
        #    nn.Linear(4*d_hidden, d_hidden)
        #)

        self.FFN = SwiGLU(d_hidden)
        #init the weights
        self.apply(self._init_weights)
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
    
    def forward(self, X):
        X_norm = self.rms_norm_ln(X)

        attn = self.attention_layer(X_norm)

        X_1 = attn+X
        X_norm_1 = self.rms_norm_ffn(X_1)

        return self.FFN(X_norm_1) + X_1


if __name__ == "__main__":

    d_hidden = 1024
    n_heads = 32
    n_groups = 8

    X = torch.randn(8,1,d_hidden)

    layer = transformer(d_hidden = d_hidden, n_heads = n_heads, n_groups = n_groups)

    output = layer(X)

    
    print(f"Our input X of shape {X.shape} gives output {output} of dimensions {output.shape}")