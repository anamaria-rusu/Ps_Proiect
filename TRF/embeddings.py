import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model) # seq_len = nr patchuri
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # unsqueeze(1) => shape (max_seq_len, 1)
        # calc numaratorul 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # => shape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe) # setam pe ca "parametru neantrenabil"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1) #nr de patchuri
        
        x = x + self.pe[:, :seq_len, :] # adaugam pozitie pt fiecare patch
        
        return self.dropout(x)



class PatchEmbedding(nn.Module):
    """
    input:  (B, L, C)
    output: (B*C, N, D) dupa proiectia liniara + PE
    B = batch_size
    L = lungimea ferestrei de intrare
    C = numar features (Canale)
    N = numarul de patch-uri = floor((L - patch_len)/stride) + 2
    D = d_model
    """

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert patch_len > 0 and stride > 0 # verificare input valid
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model) # proiectia liniara patch_len -> d_model
        self.dropout = nn.Dropout(dropout)
        self.pe = PositionalEncoding(d_model=d_model, max_seq_len=5000, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Intrare: x: (B, L, C) seria de timp
        Iesire: patches_emb: (B*C, N, D)
        """
        B, L, C = x.shape
        #adaugam padding cu ultima val
        last_value = x[:, -1:, :] #ultima valoare (B, 1, C)
        x = torch.cat([x, last_value.repeat(1, self.stride, 1)], dim=1)
        
        # permute la (B, C, L) ptc unfold se face pe dim 2
        x_bcL = x.permute(0, 2, 1)
        x_patches = x_bcL.unfold(dimension=2, size=self.patch_len, step=self.stride) #extragem patch-urile prin fereastra glisanta
        #=> (B, C, N, patch_len)
        B, C, N, P = x_patches.shape

        # proiectia liniara patch_len -> d_model
        patches_emb = self.proj(x_patches)  # (B, C, N, D)
        patches_emb = self.dropout(patches_emb)

        # channel-independent: rearanjam in (B*C, N, D) ptc trasformerul le trateaza ca secvente independente
        patches_emb = patches_emb.reshape(B * C, N, self.d_model)
        patches_emb = self.pe(patches_emb)  #adaugam positional encoding
        return patches_emb