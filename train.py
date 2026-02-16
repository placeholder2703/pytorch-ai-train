import time
from datetime import datetime
train_start = datetime.now()
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
# --------------------
# CONFIG
# --------------------
# Model config
N_EMBD = 384
N_LAYER = 4
SEQ_LEN = 256
DEVICE = "cpu"

# Train settings
BATCH_SIZE = 2

# Directories
DATA_PATH = "data.txt"
TOKENIZER_PATH = "tokens.json"
SAVE_PATH = "model.pt"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR="logs"
os.makedirs(CHECKPOINT_DIR,exist_ok=True)
os.makedirs(LOG_DIR,exist_ok=True)

# LR scheduler
TOTAL_STEPS = 40000
WARMUP_STEPS = 1000
MAX_LR = 2e-4
MIN_LR = 1e-5

# Extras
BUFFER_MAX = 20000
BUFFER_MIN = 2000
doLog = True
# --------------------
# LOG
# --------------------
if doLog:
	log_path=os.path.join(LOG_DIR,f"latest.log")
	log_file=open(log_path,"w",encoding="utf-8")
def log(showtime, msg):
	if showtime:
		line=f"[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}"
	else:
		line=f"{msg}"
	print(msg)
	if doLog:
		if log_file:
			log_file.write(line+"\n")
			log_file.flush()
log(True, f"[CONFIG] Embeddings: {N_EMBD}")
log(True, f"[CONFIG] Layer(s): {N_LAYER}")
log(True, f"[CONFIG] Context length: {SEQ_LEN} tokens")
log(True, f"[CONFIG] Max buffer size: {BUFFER_MAX}")
log(True, f"[CONFIG] Min buffer size: {BUFFER_MIN}")
# --------------------
# TOKENIZER
# --------------------
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()
log(True, f"[TOKENIZER] Vocabulary size: {VOCAB_SIZE}")
def encode(s):
	return tokenizer.encode(s).ids
def decode(ids):
	return tokenizer.decode(ids)
# --------------------
# STREAMING
# --------------------
def token_stream(path):
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			ids = encode(line)
			for t in ids:
				yield t
stream = token_stream(DATA_PATH)
token_buffer = []
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
		T = idx.size(1)
		pos = torch.arange(T, device=idx.device)
		x = self.token_emb(idx) + self.pos_emb(pos)
		for blk in self.blocks:
			x = blk(x)
		x = self.ln_f(x)
		return self.head(x)
model = TinyGPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=MIN_LR)
loss_fn = nn.CrossEntropyLoss()
# --------------------
# RESUME
# --------------------
step = 0
if os.path.exists(SAVE_PATH):
	ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
	model.load_state_dict(ckpt["model_state"])
	optimizer.load_state_dict(ckpt["optimizer_state"])
	step = ckpt["step"]
	log(True, f"[RESUME] Resumed from step {step}")
log_start=step
# --------------------
# TRAIN
# --------------------
session_steps = 0
saves = 0
ckpts = 0
stime_sum = 0
loss_sum = 0.0
loss_count = 0
lowest_loss = 15
ema_loss = None
lowest_ema = 15
EMA_ALPHA = 0.01
try:
	while True:
		# Refill buffer only when needed
		if len(token_buffer) < BUFFER_MIN:
			while len(token_buffer) < BUFFER_MAX:
				try:
					token_buffer.append(next(stream))
				except StopIteration:
					stream = token_stream(DATA_PATH)
					token_buffer.append(next(stream))
			log(True, f"[STREAM] Refilled tokens buffer")
		chunks = []
		for _ in range(BATCH_SIZE):
			chunk = token_buffer[:SEQ_LEN + 1]
			token_buffer = token_buffer[SEQ_LEN:]
			chunks.append(chunk)
		x = torch.tensor([c[:-1] for c in chunks], dtype=torch.long)
		y = torch.tensor([c[1:] for c in chunks], dtype=torch.long)
		start = time.time()
		logits = model(x)
		loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		if step < WARMUP_STEPS:
			lr = 1e-6 + (MAX_LR - 1e-6) * (step / WARMUP_STEPS)
		else:
			progress = min((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), 1.0)
			cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
			lr = MIN_LR + (MAX_LR - MIN_LR) * cosine.item()
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr
		optimizer.step()
		step_time = (time.time() - start) * 1000
		stime_sum += step_time
		loss_sum += loss.item()
		loss_count += 1
		if loss.item() < lowest_loss:
			lowest_loss = loss.item()
		if ema_loss is None:
			ema_loss = loss.item()
		else:
			ema_loss = (1 - EMA_ALPHA) * ema_loss + EMA_ALPHA * loss.item()
		if ema_loss < lowest_ema:
			lowest_ema = ema_loss
		log(True, f"[TRAIN] STEP {step} | LOSS {loss.item():.4f} | EMA {ema_loss:.4f} ({step_time:.0f}ms)")
		if step % 100 == 0 and step > 0:
			torch.save({
				"step": step,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
			}, SAVE_PATH)
			log(True, f"[TRAIN] Model saved to {SAVE_PATH}")
			saves += 1
		if step % 5000 == 0 and step > 0:
			ckpt_path = os.path.join(CHECKPOINT_DIR, f"{step}.pt")
			torch.save({
				"step": step,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
			}, ckpt_path)
			log(True, f"[TRAIN] Saved checkpoint {step} to {ckpt_path}")
			ckpts += 1
		session_steps += 1
		step += 1
except KeyboardInterrupt:
	final_log_path=os.path.join(LOG_DIR,f"{log_start}-{step}.log") if doLog else "Logging is disabled"
	avg_loss = loss_sum / loss_count if loss_count > 0 else float("nan")
	train_end = datetime.now()
	train_time = int((train_end - train_start).total_seconds())
	d = train_time // 86400
	h = (train_time % 86400) // 3600
	m = (train_time % 3600) // 60
	s = train_time % 60
	log(False, "\nKeyboardInterrupt caught")
	log(False, "==========STATS==========")
	log(False, f"Steps trained: {session_steps}")
	log(False, f"Time elapsed: {d:02}:{h:02}:{m:02}:{s:02}")
	log(False, f"Average step time: {(stime_sum / session_steps):.0f} ms")
	log(False, f"Average loss: {avg_loss:.5f}")
	log(False, f"Lowest loss reached: {lowest_loss:.5f}")
	log(False, f"Final EMA loss: {ema_loss:.5f}")
	log(False, f"Lowest EMA loss: {lowest_ema:.5f}")
	log(False, f"Checkpoints generated: {ckpts}")
	log(False, f"Saves written: {saves}")
	log(False, f"Log path: {final_log_path}")
	log(False, "=======END OF STATS======")
	torch.save({
		"step": step,
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
	}, SAVE_PATH)
	if doLog:
		log_file.close()
		os.rename(log_path,final_log_path)
	del model
	del optimizer
	del token_buffer
	torch.cuda.empty_cache()
	input("Press enter to exit...")
