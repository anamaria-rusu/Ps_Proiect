import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, num_heads, seq_len, d_k)
            K: (batch_size, num_heads, seq_len, d_k)
            V: (batch_size, num_heads, seq_len, d_v)
            mask: (batch_size, 1, 1, seq_len)
        Outputs:
            output: (batch_size, num_heads, seq_len, d_v)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        
        #calculare scoruri: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        #aplicam mask (pentru a ignora padding-urile)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        #aplicare softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        #inmultim matricea de atentie cu V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1,....,head_h)W^O
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: dimensiunea modelului (embedding size)
            num_heads: numarul de attention heads
            dropout: probabilitate dropout
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model trebuie sa fie divizibil cu num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        #Linear layers pentru Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        #W_0 proiectia finala 
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """
        Imparte embedding-ul in num_heads subspatii
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        x_view = x.view(batch_size, seq_len, self.num_heads, self.d_k) # (B, L, h, d_k) imparte d_model in heads
        x_transpose = x_view.transpose(1, 2) # (B, h, L, d_k)
        return x_transpose
    
    def combine_heads(self, x):
        """
        Concateneaza head-urile
        Args:
            x: (batch_size, num_heads, seq_len, d_k)
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None): #pentru modularitate, Q,K,V pot fi diferite de X_final
        """
        Args:
            Q: (batch_size, seq_len_q, d_model)
            K: (batch_size, seq_len_k, d_model)
            V: (batch_size, seq_len_v, d_model)
            mask: (batch_size, 1, seq_len, seq_len)
        
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q.size(0)
        
        #proiectiile liniare X * W^Q, W^K, W^V
        Q = self.W_q(Q) # (batch_size, seq_len_q, d_model)
        K = self.W_k(K) # (batch_size, seq_len_k, d_model)
        V = self.W_v(V) # (batch_size, seq_len_v, d_model)
        
        #impartim in heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)
        
        #aplicam Scaled Dot-Product Attention pe fiecare head
        attn_output, attention_weights = self.attention(Q, K, V, mask) #se executa simultan pe toate head-urile
        
        # Concateneaza head-urile
        attn_output = self.combine_heads(attn_output)  # (batch_size, seq_len_q, d_model)
        
        # Proiectia finala
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights                            
