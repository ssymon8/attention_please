import torch
import torch.nn as nn
from attention import Attention_layer

"""La Multi-Head Attention c'est juste des layers d'attention plus petites 
stackées pour former une grosse layer"""

class MHA_layer(nn.Module):
    def __init__(self, d_hidden: int = 1024, n_heads: int = 32, causal: bool = True):
        super().__init__()
        assert d_hidden%n_heads == 0

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_heads = d_hidden// n_heads

        self.attention_heads = nn.ModuleList(Attention_layer(d_hidden = self.d_heads) for _ in range(n_heads))
    
        self.out_proj = nn.Linear(d_hidden, d_hidden)

    def forward(self, x)-> torch.Tensor:
        head_outputs = []
        for head in self.attention_heads:
            out = head(x)
            head_outputs.append(out)
        
        final_out = torch.cat(head_outputs, dim= -1)

        return self.out_proj(final_out)

def main():
    layer=MHA_layer(d_hidden = 256,n_heads = 8, causal = True)

    x= torch.randn(8,1,32)
    y= layer(x)
    print(y)

if __name__ == "__main__":
    main()
