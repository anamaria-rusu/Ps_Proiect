"""
Implementare Encoder pentru arhitectura Transformer
"""
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layers import PositionWiseFeedForward, ResidualConnection


class EncoderLayer(nn.Module):
    """
    Layer de Encoder: Multi-Head Self-Attention + Feed-Forward
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: dimensiunea modelului (embedding size)
            num_heads: numarul de attention heads
            d_ff: dimensiunea feed-forward network
            dropout: probabilitate dropout
        """
        super(EncoderLayer, self).__init__()
        
        #MHSA
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        #FFN
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        #residual + normalizare
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len) mask pentru padding
        
        Returns:
            (batch_size, seq_len, d_model)

        """
        #MHSA -> Add & Norm
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, mask)[0]) 
        
        #FFN -> Add & Norm
        x = self.residual2(x, self.feed_forward)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack de N encoder layers
    (MHSA -> Add & Norm -> FFN -> Add & Norm) x N
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers: numarul de encoder layers
            d_model: dimensiunea modelului
            num_heads: numarul de attention heads
            d_ff: dimensiunea feed-forward network
            dropout: probabilitate dropout
        """
        super(TransformerEncoder, self).__init__()
        
        #stack de encodere
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len)
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        #trecem prin fiecare encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
