from pathlib import Path
import time, torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from seora.config import SeoraConfig
from seora.model import SeoraAI
from tokenizer import encode, decode

app = FastAPI(title="SeoraAI", version="0.1.0")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def load_model():
    global model
    ckpt = Path("checkpoints/seora-ai-1.pt")
    if not ckpt.exists():
        return None
    data = torch.load(ckpt, map_location=device)
    cfg = SeoraConfig(**data["config"])
    model = SeoraAI(cfg).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "seora-ai-1"
    messages: list[Message]
    max_tokens: int = 120
    temperature: float = 0.8

@app.on_event("startup")
def startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": device}

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id":"seora-ai-1","object":"model","owned_by":"Losh"}]}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(503, "Model checkpoint not found. Run: python train.py")
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    x = torch.tensor([encode(prompt)[-model.cfg.block_size:]], dtype=torch.long, device=device)
    start = time.perf_counter()
    out = model.generate(x, max_new_tokens=min(req.max_tokens, 256), temperature=req.temperature)
    elapsed = time.perf_counter() - start
    text = decode(out[0].tolist())
    completion = text[len(prompt):]
    return {
        "id": f"seora-{int(time.time())}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index":0,"message":{"role":"assistant","content":completion},"finish_reason":"length"}],
        "seora_meta":{"latency_seconds":round(elapsed,3)}
    }
