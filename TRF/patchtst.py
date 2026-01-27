"""
PatchTST: A Time Series is Worth 64 Words (Long-Term Forecasting with Transformers)
"""
import torch
import torch.nn as nn
from embeddings import PatchEmbedding, PositionalEncoding
from encoder import TransformerEncoder
from layers import ReversibleInstanceNormalization


class PatchTST(nn.Module):
    """
    1. Input: (B, L, C) - seria de timp cu batch size B, lungime L, canale C
    2. Patching + Linear Projection: (B, L, C) -> (B*C, N, D)
    3. Positional Encoding
    4. Transformer Encoder: (B*C, N, D) -> (B*C, N, D)
    5. Flatten: (B*C, N, D) -> (B*C, N*D)
    6. Prediction Head: (B*C, N*D) -> (B*C, pred_len)
    7. Reshape: (B*C, pred_len) -> (B, pred_len, C)
    """
    
    def __init__(self, 
                 seq_len: int,
                 pred_len: int,
                 in_channels: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 use_revin: bool = True):
        """
        Args:
            seq_len: lungimea seriei de intrare (L)
            pred_len: lungimea predictiei
            in_channels: numarul de caracteristici/canale (C)
            patch_len: dimensiunea patch-ului
            stride: pasul patch-urilor
            d_model: dimensiunea modelului (embedding-urilor)
            num_heads: numarul de attention heads
            num_layers: numarul de encoder layers
            d_ff: dimensiunea FFN-ului
            dropout: probabilitate dropout
            use_revin: daca True, foloseste RevIN pentru normalizare
        """
        super(PatchTST, self).__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.d_model = d_model
        self.use_revin = use_revin
        
        # Calculeaza numarul de patch-uri
        # N = floor((L + padding - patch_len) / stride) + 1
        # Cu padding = stride, avem N ≈ floor(L / stride)
        self.num_patches = (seq_len + stride - patch_len) // stride + 1
        
        #RevIN
        if self.use_revin:
            self.revin = ReversibleInstanceNormalization(num_features=in_channels, affine=True)
        
        #Patch Embedding (patching + linear projection + PE)
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)
        
        #Transformer Encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        #Flatten + Prediction Head
        self.flatten = nn.Flatten(start_dim=1)  # (B*C, N, D) -> (B*C, N*D)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)
        
        #xavier init
        self._init_weights()
    
    def _init_weights(self):
        #Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, in_channels)
            mask: pentru padding
        
        Returns:
            output: (batch_size, pred_len, in_channels)
        """
        B, L, C = x.shape
        
        #Normalizare RevIN
        if self.use_revin:
            x, (mean, var) = self.revin(x, return_stats=True)
        
        #Patching + Linear Projection + Positional Encoding (B, L, C) -> (B*C, N, D)
        x_emb = self.patch_embedding(x)
        
        #Transformer Encoder (B*C, N, D) -> (B*C, N, D)
        x_enc = self.encoder(x_emb, mask)
        
        #Flatten (B*C, N, D) -> (B*C, N*D)
        x_flat = self.flatten(x_enc)
        x_flat = self.dropout(x_flat)
        
        #Prediction Head (B*C, N*D) -> (B*C, pred_len)
        forecast = self.head(x_flat)
        
        #reshape la (B, pred_len, C)
        #(B*C, pred_len) -> (B, C, pred_len) -> (B, pred_len, C)
        forecast = forecast.reshape(B, C, self.pred_len)
        forecast = forecast.permute(0, 2, 1) #(B, pred_len, C)
        
        #denormalizare output
        if self.use_revin:
            forecast = self.revin.inverse(forecast, mean, var)
        
        return forecast


class PatchTSTForecasting(nn.Module):
    
    def __init__(self, config: dict):
        """
        Args:
            config: dictionar cu parametri
            - seq_len, pred_len, in_channels
            - patch_len, stride
            - d_model, num_heads, num_layers, d_ff
            - dropout
        """
        super(PatchTSTForecasting, self).__init__()
        
        self.model = PatchTST(
            seq_len=config.get('seq_len', 336),
            pred_len=config.get('pred_len', 96),
            in_channels=config.get('in_channels', 1),
            patch_len=config.get('patch_len', 16),
            stride=config.get('stride', 8),
            d_model=config.get('d_model', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 2048),
            dropout=config.get('dropout', 0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
