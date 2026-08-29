from pathlib import Path
import torch
from torch.optim import AdamW
from seora.config import SeoraConfig
from seora.model import SeoraAI
from tokenizer import encode

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = SeoraConfig()
model = SeoraAI(cfg).to(device)

text = Path("data/train.txt").read_text(encoding="utf-8")
ids = torch.tensor(encode(text), dtype=torch.long)
if len(ids) < cfg.block_size + 2:
    ids = ids.repeat((cfg.block_size + 2) // len(ids) + 1)

optimizer = AdamW(model.parameters(), lr=3e-4)
batch_size = 8
steps = 500

for step in range(steps):
    starts = torch.randint(0, len(ids) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([ids[s:s+cfg.block_size] for s in starts]).to(device)
    y = torch.stack([ids[s+1:s+cfg.block_size+1] for s in starts]).to(device)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 50 == 0:
        print(f"step={step} loss={loss.item():.4f}")

Path("checkpoints").mkdir(exist_ok=True)
torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, "checkpoints/seora-ai-1.pt")
print("Saved checkpoints/seora-ai-1.pt")
