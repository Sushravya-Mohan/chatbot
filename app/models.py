import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SimpleTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.size()
        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.fc_out(concat)
        return output, attn_weights

class ChatbotModel(nn.Module):
    def __init__(self, d_model=64, num_heads=4, vocab_size=1000):
        super(ChatbotModel, self).__init__()
        self.d_model = d_model
        self.transformer = SimpleTransformerBlock(d_model, num_heads)
        self.response_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        transformer_output, _ = self.transformer(x)
        logits = self.response_head(transformer_output)
        return logits
