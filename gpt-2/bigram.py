import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters

# when decreasing learning rate, increase number of iterations
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 300
eval_iters = 200
lr = 3e-4
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------

torch.manual_seed(1337)

with open(os.path.join(".", "data", "input.txt"), "r", encoding="utf-8") as f:
    text = f.read()

# create unique characters
unique_chars = list(set(text))
vocab_size = len(unique_chars)

# create mapping from characters
stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


# train test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    data = train_data if split == "train" else val_data

    starting_idx = torch.randint(low=0, high=len(
        data) - block_size, size=(batch_size, ))

    x = torch.stack([data[id:id+block_size] for id in starting_idx])
    y = torch.stack([data[id + 1:id + 1 + block_size] for id in starting_idx])

    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

# model


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size)
        self.key = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.register_buffer("tril", torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embeddings_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embeddings_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeds = self.token_embeddings_table(idx)
        pos_embeds = self.pos_embeddings_table(torch.arange(T, device=device))

        x = token_embeds + pos_embeds
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # get the last block size token
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # predict
    logits, loss = model(xb, yb)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), "./model/gpt2.pt")
