import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_hidden : int, d_ff : int = None, multiple : int = 64): #en général, d_ff = 8/3d_model pour garder un budget de paramètre comparable à un FFN classique
        super().__init__()
        
        if d_ff is None:
            d_ff = int(2 * (4*d_hidden)/3)
            # on arrondit à un multiple de 64 pour optimiser les alignements CPU/GPU
            d_ff = multiple * ((d_ff+multiple - 1)//multiple)

        assert d_ff%multiple == 0, f"d_ff = {d_ff} doit idéalement être un multiple de {multiple} svp"

        self.w1 = nn.Linear(d_hidden, d_ff, bias= False)
        self.v = nn.Linear(d_hidden, d_ff, bias= False)
        self.w2 = nn.Linear(d_ff, d_hidden, bias= False)
    
    def forward(self, X):
        assert X.shape[-1] == self.w1.in_features, "Dimension d_model incohérente"
        #print("SWIGLU PASSED")
        return self.w2(F.silu(self.w1(X)) * self.v(X))
        


if __name__ == "__main__":

    d_hidden = 1024

    FFN = SwiGLU(d_hidden)

    X = torch.randn(8,1,d_hidden)

    output = FFN(X)

    
    print(f"Our input X of shape {X.shape} gives output {output} of dimensions {output.shape}")
