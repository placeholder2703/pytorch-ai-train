import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
# --------------------
# CONFIG (must match training)
# --------------------
N_EMBD = 384
N_LAYER = 4
SEQ_LEN = 256
DEVICE = "cpu"
TOKENIZER_PATH = "tokens.json"
MODEL_PATH = "model.pt"
# --------------------
# TOKENIZER
# --------------------
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()
def encode(s):
    return tokenizer.encode(s).ids
def decode(ids):
    return tokenizer.decode(ids)
# --------------------
# MODEL
# --------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
        )
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)                 # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)    # each (B, T, C)
        att = (q @ k.transpose(-2, -1)) / (C ** 0.5)  # (B, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v                     # (B, T, C)
        return self.proj(out)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(SEQ_LEN, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)
    def forward(self, idx):
        T = idx.size(0)
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)
model = TinyGPT().to(DEVICE)
# --------------------
# LOAD WEIGHTS
# --------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("[LOAD] Loaded model at step", ckpt["step"])
# --------------------
# GENERATION
# --------------------
def generate(prompt, length=100, temperature=0.8):
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(length):
        # keep last SEQ_LEN tokens (on token dimension!)
        idx_cond = idx[:, -SEQ_LEN:]
        logits = model(idx_cond)          # (1, T, VOCAB)
        logits = logits[:, -1, :]         # (1, VOCAB)
        probs = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1)  # (1, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return decode(idx[0].tolist())
# --------------------
# INTERACTIVE LOOP
# --------------------
while True:
    prompt = input("[GENERATE] START > ")
    length = int(input("[GENERATE] LENGTH > "))
    temperature = float(input("[GENERATE] TEMPATURE > "))
    print("[GENERATE]",generate(prompt, length, temperature))
    print()
