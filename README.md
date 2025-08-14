# GPT2 from scratch for text gen and classication


This repository contains two complementary pieces:

1. **A GPT‑2–based SMS spam classifier** that fine‑tunes a pretrained GPT‑2 checkpoint with a light‑weight adaptation strategy (LoRA‑style) and a small classification head.
2. **A complete GPT‑2 text generation pipeline** (model from scratch, weight loader, tokenizer utilities, and sampler) for easy experimentation and demos.

> If you’re evaluating the repo quickly: run `python Classifier_model.py` to fine‑tune and evaluate the classifier; run the **Combined Pipeline** script (see below) to generate text from GPT‑2.

---

## Contents
- [Why this project?](#why-this-project)
- [Project architecture](#project-architecture)
- [LoRA in 30 seconds](#lora-in-30-seconds)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Quickstart: Train the spam classifier](#quickstart-train-the-spam-classifier)
- [Inference: Classify new messages](#inference-classify-new-messages)
- [Combined GPT‑2 training & inference pipeline](#combined-gpt-2-training--inference-pipeline)
- [Configuration knobs](#configuration-knobs)
- [Dataset notes](#dataset-notes)
- [Troubleshooting & FAQs](#troubleshooting--faqs)
- [Roadmap](#roadmap)
- [Attribution & License](#attribution--license)

---

## Why this project?
- **Practical**: shows how to take a pretrained LLM (GPT‑2) and adapt it to a downstream task (spam/ham classification) with minimal compute.
- **Educational**: includes a from‑scratch GPT‑2 implementation and a tiny, readable sampler so you can *see* what’s going on under the hood.
- **Portable**: works on CPU; runs much faster on GPU; uses small GPT‑2 (“124M”) weights by default for accessibility.

---

## Project architecture

### 1) Spam classifier (fine‑tuning GPT‑2)
- Downloads the **SMS Spam Collection** dataset.
- Balances classes (ham vs spam).
- Tokenizes texts with **tiktoken (GPT‑2 BPE)**.
- Loads GPT‑2 weights.
- Replaces/augments parts of the model with a **classification head** and optional **LoRA‑style adapters**.
- Trains & evaluates; saves **loss/accuracy** plots as PDFs.

### 2) Combined GPT‑2 pipeline (text generation)
- Minimal implementation of the GPT‑2 blocks (attention, MLP, norms).
- Loads official GPT‑2 weights (OpenAI public mirror).
- Generates text from a prompt using temperature & top‑k sampling.

---

## LoRA in 30 seconds
LoRA (Low‑Rank Adaptation) injects small trainable matrices **A** and **B** into a frozen weight **W**, so the effective weight during training becomes:

\[\ W_\text{eff} = W + \frac{\alpha}{r}\, A B \]

- **r** = rank (e.g., 8), **α** = scaling.
- Train only **A**/**B** (and a small head), **freeze** the big pretrained weights → **much fewer trainable params**, lower memory, faster fine‑tuning.

> **Note:** The provided `Linear_LORA` module is a compact adaptation layer. If you want a canonical LoRA (where `B` maps from `rank → out_dim`), see **Tips** in the FAQ below.

---

## Repository layout

```
.
├── Classifier_model.py            # Train/evaluate the spam classifier (LoRA-style adapters + classification head)
├── gpt_download.py                # Robust GPT‑2 checkpoint downloader/loader (TensorFlow ckpt reader)
├── trainmodel.py                  # GPT‑2 weight loading helpers (alt path)
├── Unlabeled_data/                # Expected to contain: unlabeled_model.py (GPTModel, GPTConfig, load_weights_into_gpt, …)
│   ├── gpt_download.py            # (Optional) if you keep utilities under this package
│   └── unlabeled_model.py         # (If missing, see Combined Pipeline below)
└── README.md
```

> **If you don’t have `Unlabeled_data/unlabeled_model.py`**, you can still run the **Combined GPT‑2 Pipeline** (which defines the model inline), or move those definitions into `Unlabeled_data/unlabeled_model.py` as expected by `Classifier_model.py`.

---

## Setup

### 1) Python
- Python **3.9+** recommended.

### 2) Install dependencies
```bash
# (optional) create a venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch tensorflow numpy pandas matplotlib tqdm tiktoken
```

> **Why TensorFlow?** Only used by `gpt_download.py` to read original GPT‑2 TF checkpoints. PyTorch does the modeling/training.  
> You can remove the TF dependency if you convert weights once and ship a PyTorch‑native state dict.

### 3) GPU (optional but recommended)
- CUDA‑enabled PyTorch speeds up training/inference.
- Reduce `batch_size` if you hit **CUDA OOM**; the default is small for approachability.

---

## Quickstart: Train the spam classifier

```bash
python Classifier_model.py
```

What it does:
- Downloads the SMS Spam dataset (with a **backup URL**).
- Creates `train.csv`, `validation.csv`, `test.csv` (balanced classes).
- Loads GPT‑2 (“124M”) weights.
- Adds a **2‑class output head**, freezes most weights, and (optionally) uses LoRA‑style adapters.
- Trains for `num_epochs` (default 5), prints metrics.
- Saves plots: `loss-plot.pdf` and `Accuracy-plot.pdf`.

**Key output (example):**
```
Epochs 1 (Step 000050): Train loss: 0.485, Validation loss: 0.432
Training Accuracy: 89.00% | Validation Accuracy: 87.50%
```

### Enable/disable LoRA adapters
In `Classifier_model.py`, the replacement is controlled by the `alternative` flag:
```python
replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, alternative=True)
```
> Set `alternative=True` to actually insert `Linear_LORA` layers. With `False`, no layers are replaced (you’ll fine‑tune the last block + final norm + `out_head`).

---

## Inference: Classify new messages

```python
from Classifier_model import classify_review
import tiktoken, torch

# Load the same GPTModel & tokenizer you trained with
# (If you used Unlabeled_data/unlabeled_model.py)
from Unlabeled_data.unlabeled_model import GPTModel, GPTConfig, load_weights_into_gpt
from Unlabeled_data.gpt_download import download_and_load_gpt2

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model & load weights (match training config)
model = GPTModel(GPTConfig(vocab_size=50257, context_length=1024, emb_dim=768, n_heads=12, n_layers=12, drop_rate=0.0, qkv_bias=True))
_, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
load_weights_into_gpt(model, params)
model.to(device)

text = "You won $1000! Claim now."
print(classify_review(text, model, tokenizer, device, max_length=128))  # -> "spam" or "not spam"
```

**Notes:**
- `classify_review` expects `max_length` to be provided. Use `train_dataset.max_length` or a safe number like 128–256.
- The classifier uses the **last token’s logits** from the model head for the 2‑class decision.

---

## Combined GPT‑2 training & inference pipeline

If you want a *single, self‑contained* script that **defines** the GPT‑2 model, **downloads** weights, and **generates** text, use the following as `combined_gpt2_pipeline.py` (this is exactly what the project includes/accepts):

<details>
<summary><strong>Show code (click to expand)</strong></summary>

```python
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
```
</details>

Run it:
```bash
python combined_gpt2_pipeline.py
```

---

## Configuration knobs

| Setting | Where | Purpose | Notes |
|---|---|---|---|
| `CHOOSE_MODEL` | `Classifier_model.py` | Select GPT‑2 size | Default “gpt2-small (124M)” |
| `batch_size` | `Classifier_model.py` | Memory/throughput tradeoff | Lower for small GPUs/CPU |
| `num_epochs` | `Classifier_model.py` | Training duration | Default 5 |
| `lora_rank`, `lora_alpha`, `lora_dropout` | `Classifier_model.py` | LoRA adapter size/scaling | Set `alternative=True` to enable replacement |
| `context_length` | Model config | Max tokens fed to GPT‑2 | Must match position embeddings |
| `eval_freq`, `eval_iter` | Trainer | How often to compute/average metrics | Useful for speed/variance |
| `drop_rate`, `qkv_bias` | Model config | Architectural variants | Defaults match GPT‑2 small |

---

## Dataset notes

- Source: **[UCI SMS Spam Collection]** – automatically downloaded as a ZIP; a backup URL is included.
- Files created: `train.csv`, `validation.csv`, `test.csv` with columns: `Label` (0=ham, 1=spam), `Text` (message).
- The code **balances** classes by down‑sampling the majority class for a more stable classifier.

---

## Troubleshooting & FAQs

**Q: I don’t see any LoRA parameters training.**  
A: Ensure you call `replace_linear_with_lora(..., alternative=True)`. With `False`, no layers are replaced. Also verify you unfreeze `out_head` and any layers you *do* want to train.

**Q: CUDA out of memory (OOM).**  
A: Reduce `batch_size` (e.g., 4 → 2 → 1), shorten `max_length`, or run on CPU (slower). Close other GPU processes.

**Q: TensorFlow errors when loading GPT‑2 weights.**  
A: TensorFlow is used **only** to read the original checkpoint. Ensure your TF install matches your Python version. Alternatively, pre‑convert weights to PyTorch and skip TF.

**Q: `ModuleNotFoundError: Unlabeled_data.unlabeled_model`.**  
A: Either: (1) add `Unlabeled_data/unlabeled_model.py` (with `GPTModel`, `GPTConfig`, `load_weights_into_gpt`), or (2) use the **Combined Pipeline** script which defines these inline.

**Q: `classify_review` complains about `max_length`.**  
A: Pass a value explicitly (e.g., `max_length=128`). A safe default is your training dataset’s `train_dataset.max_length`.

**Tip (canonical LoRA):** In a standard LoRA linear, `A: in_dim → rank` and `B: rank → out_dim`. The provided `Linear_LORA` is minimal and uses `lora_b = Linear(in_dim, out_dim)`; to mirror canonical LoRA, change it to `Linear(rank, out_dim)` and adjust the forward pass accordingly.

---

## Roadmap
- ✅ Balanced dataset, quick training loop, plots
- ✅ Combined GPT‑2 generation pipeline
- 🔲 Save/Load fine‑tuned classifier weights to disk
- 🔲 Proper canonical LoRA blocks for attention/MLP projections
- 🔲 CLI flags (`argparse`) for hyperparameters
- 🔲 PyTorch‑native GPT‑2 weight loader (remove TensorFlow)

---

## Attribution & License

Parts of this repository adapt utilities from **Sebastian Raschka’s “LLMs from Scratch”** (Apache 2.0).  
- Book: *Build a Large Language Model From Scratch*  
- Code: https://github.com/rasbt/LLMs-from-scratch

Dataset: **SMS Spam Collection** (UCI Machine Learning Repository). Respect dataset license/terms.

Unless noted otherwise, new code in this repository is provided under your chosen license. If you re‑publish, please keep attributions for third‑party code.

---

### Citation (optional)
If you use this in academic or blog posts, a short citation helps others find the code:

> GPT‑2 SMS Spam Classifier (LoRA) + Combined GPT‑2 Pipeline, 2025. https://github.com/<your‑username>/<your‑repo>

---

**Happy hacking!** If anything feels unclear, open an issue or ping the maintainer.
