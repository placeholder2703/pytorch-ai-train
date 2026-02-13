from tokenizers import Tokenizer, models, trainers, pre_tokenizers

DATA_PATH = "data.txt"
TOKENIZER_PATH = "tokens.json"
VOCAB_SIZE = 32768

def line_iterator(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>"]
)

tokenizer.train_from_iterator(line_iterator(DATA_PATH), trainer)

tokenizer.save(TOKENIZER_PATH)
print("[TOKENIZER] Saved tokenizer to", TOKENIZER_PATH)
print("[TOKENIZER] Vocabulary size:", tokenizer.get_vocab_size())
