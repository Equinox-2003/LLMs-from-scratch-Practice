# 前面章节的一些code

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # Tokenizes the entire text
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            x = token_ids[i:i+max_length]
            y = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(x))
            self.target_ids.append(torch.tensor(y))
    
    def __len__(self): 
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,    # we drop the last batch if it's shorter than batch_size
        num_workers=num_workers # num of GPU processes
    )
    return dataloader

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0, \
               "d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # Reduces the projection dim to match the desired output dim
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # Tensor shape: (b, num_tokens, d_out)
        ks = self.W_k(x)
        qs = self.W_q(x)
        vs = self.W_v(x)
        #We implicitly split the matrix by adding a num_heads dimension. 
        #Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        ks = ks.view(b, num_tokens, self.num_heads, self.head_dim)
        vs = vs.view(b, num_tokens, self.num_heads, self.head_dim)
        qs = qs.view(b, num_tokens, self.num_heads, self.head_dim)
        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        ks = ks.transpose(1, 2)
        qs = qs.transpose(1, 2)
        vs = vs.transpose(1, 2)
        # Computes dot product for each head
        attn_scores = qs @ ks.transpose(2, 3)
        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Uses the mask to fill attention scores
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / ks.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ vs).transpose(1, 2)
        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        # Adds an optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec