"""
Implementare layere auxiliare pentru Transformer
"""
import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: dimensiunea inputului/outputului
            d_ff: dimensiunea hidden layer (de obicei 4 * d_model)
            dropout: probabilitate dropout
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        #x -> linear1 -> relu -> dropout -> linear2 -> dropout
        output = self.linear1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        return output


class LayerNorm(nn.Module):
    """
    Layer Normalization: Normalizeaza de-a lungul dimensiunii feature-urilor
    """
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: dimensiunea featurelor
            eps: valoare mica pentru stabilitate numerica
        """
        super(LayerNorm, self).__init__()

        #parametrii invatabili gamma si beta
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """
    Conexiune reziduala + Layer Normalization (Add & Norm)
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: dimensiunea modelului
            dropout: probabilitate dropout
        """
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        LayerNorm(x + Dropout(Sublayer(x)))
        
        Args:
            x: (batch_size, seq_len, d_model)
            sublayer: MHSA sau FFN
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        return x + self.dropout(sublayer(self.norm(x)))




class ReversibleInstanceNormalization(nn.Module):
   
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        """
        Args:
            num_features: numarul de caracteristici (canale)
            eps: valoare mica pentru stabilitate numerica
            momentum: momentum pentru moving_mean si moving_var
            affine: daca True, invatam parametrii gamma si beta
        """
        super(ReversibleInstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features)) #la inceput nicio scalare => gamma=1 identitate
            self.beta = nn.Parameter(torch.zeros(num_features)) #la inceput niciun shift => beta=0 identitate
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        #retinem parametri de normalizare, pentru denormalizarea de la final
        self.register_buffer('running_mean', torch.zeros(num_features)) #media mobila 
        self.register_buffer('running_var', torch.ones(num_features))   #varianta mobila 
    
    def forward(self, x, return_stats=False):
        """
        Args:
            x: tensor de intrare de forma (B, L, C)
            return_stats: daca True, returneaza media/varianta pentru inversare
        
        Returns:
            x_norm: tensor normalizat (B, L, C)
            stats: (media, varianta) pentru inversare, forme (B, C)
        """
        if x.dim() != 3:
            raise ValueError("ReversibleInstanceNormalization asteapta tensori 3D (B, L, C)")

        # media si varianta pe dimensiunea temporala (seq_len) pentru fiecare sample si canal
        # forme: (B, 1, C)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)

        # normalizare
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # aplicam scalare si shift daca affine=True (broadcast pe (B, 1, C))
        if self.affine:
            gamma = self.gamma.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            x_norm = x_norm * gamma + beta

        if return_stats:
            # intoarcem (B, C) pentru mean si var
            return x_norm, (mean.squeeze(1), var.squeeze(1))

        return x_norm
    
    def inverse(self, x_norm, mean, var):
        """
        Inverseaza normalizarea pentru a obtine datele originale.
        Accepta `x_norm` de forma (B, T, C) unde T poate fi diferit de L (ex: pred_len),
        folosind media/varianta pe (B, 1, C) calculate din intrare.
        """
        if x_norm.dim() != 3:
            raise ValueError("ReversibleInstanceNormalization.inverse asteapta tensori 3D (B, T, C)")

        # inversam scalare si shift (broadcast pe (B, T, C))
        if self.affine:
            gamma = self.gamma.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            x = (x_norm - beta) / gamma
        else:
            x = x_norm

        # inversam normalizarea folosind (B, 1, C) pentru mean/var
        mean_reshaped = mean.unsqueeze(1)  # (B, 1, C)
        var_reshaped = var.unsqueeze(1)    # (B, 1, C)
        x = x * torch.sqrt(var_reshaped + self.eps) + mean_reshaped

        return x
