# -------------------------------------
# Combined GPT2 Training & Inference Pipeline
# -------------------------------------

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from dataclasses import dataclass, replace
import numpy as np

# ---------------------- Model & Config Definitions (from unlabeled_model.py) ----------------------
@dataclass
class GPTConfig:
    vocab_size : int = 50257
    context_length : int = 1024
    emb_dim : int = 768
    n_heads: int = 12
    n_layers : int = 12
    drop_rate : float = 0.1
    qkv_bias : bool = False

class MultiheadAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0
        self.num_heads = config.n_heads
        self.d_out = config.emb_dim
        self.dropout = torch.nn.Dropout(config.drop_rate)
        self.head_dim = config.emb_dim // config.n_heads

        self.W_query = torch.nn.Linear(config.emb_dim, config.emb_dim)
        self.W_key = torch.nn.Linear(config.emb_dim, config.emb_dim)
        self.W_value = torch.nn.Linear(config.emb_dim, config.emb_dim)
        self.out_proj = torch.nn.Linear(config.emb_dim, config.emb_dim)

        self.register_buffer("mask", torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        query = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_score = query @ key.transpose(2, 3)
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attention_score = attention_score.masked_fill(mask, float('-inf'))

        attention_weights = torch.softmax(attention_score / (key.shape[-1] ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)

class GELU(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=True)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GELU(),
            torch.nn.Linear(4 * config.emb_dim, config.emb_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiheadAttention(config)
        self.norm1 = LayerNorm(config.emb_dim)
        self.ff = FeedForward(config)
        self.norm2 = LayerNorm(config.emb_dim)
        self.dropout = torch.nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = x + self.dropout(self.att(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = torch.nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = torch.nn.Dropout(config.drop_rate)
        self.trf_blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = torch.nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        token_emb = self.tok_emb(idx)
        pos_embed = self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = self.drop_emb(token_emb + pos_embed)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.out_head(x)


def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    device = next(model.parameters()).device
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(float('-inf'), device=logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    from types import SimpleNamespace
    import importlib.util

    # Download and import gpt_download.py
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    spec = importlib.util.spec_from_file_location("gpt_download", filename)
    gpt_download = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpt_download)

    # Load pretrained GPT-2 weights
    settings, params = gpt_download.download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    # Build model config and model
    NEW_CONFIG = replace(GPTConfig(), emb_dim=768, n_layers=12, n_heads=12, context_length=1024, qkv_bias=True)
    model = GPTModel(NEW_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # Tokenizer and input prompt
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Every effort moves you"
    idx = text_to_tokens(prompt, tokenizer)

    # Generate output
    output_ids = generate(model, idx, max_new_tokens=25, context_size=NEW_CONFIG.context_length, top_k=50, temperature=1.5)
    print("\nGenerated text:\n", token_to_text(output_ids, tokenizer))
