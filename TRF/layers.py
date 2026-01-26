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
            x: tensor de intrare
            return_stats: daca True, returneaza media/varianta pentru inversare
        
        Returns:
            x_norm: tensor normalizat
            stats: (media, varianta) pentru inversare
        """
        # Reshape pentru a calcula pe dimensiuni instance
        if x.dim() == 3:  # (batch, seq_len, features)
            batch_size, seq_len, num_features = x.shape
            x_reshaped = x.view(batch_size * seq_len, num_features)
        else:
            x_reshaped = x
        
        #calculam media si varianta pentru fiecare instanta
        mean = x_reshaped.mean(dim=1, keepdim=True)
        var = x_reshaped.var(dim=1, unbiased=False, keepdim=True)
        #normalizam
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        
        #aplicam scalare si shift daca affine=True
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        
        #reshape la forma originala
        if x.dim() == 3:
            x_norm = x_norm.view(batch_size, seq_len, num_features)
        
        if return_stats:
            return x_norm, (mean.squeeze(1), var.squeeze(1))
        
        return x_norm
    
    def inverse(self, x_norm, mean, var):
        """
        Inverseaza normalizarea pentru a obtine datele originale
        
        Args:
            x_norm: tensor normalizat
            mean: media calculata in forward
            var: varianta calculata in forward
        
        Returns:
            x: tensor denormalizat
        """
        #reshape
        if x_norm.dim() == 3:
            batch_size, seq_len, num_features = x_norm.shape
            x_reshaped = x_norm.view(batch_size * seq_len, num_features)
        else:
            x_reshaped = x_norm
        
        #inversam scalare si shift
        if self.affine:
            x = (x_reshaped - self.beta) / self.gamma
        else:
            x = x_reshaped
        
        #inversam normalizarea
        mean_reshaped = mean.unsqueeze(1)
        var_reshaped = var.unsqueeze(1)
        x = x * torch.sqrt(var_reshaped + self.eps) + mean_reshaped
        
        #reshape la forma originala
        if x_norm.dim() == 3:
            x = x.view(batch_size, seq_len, num_features)
        
        return x
