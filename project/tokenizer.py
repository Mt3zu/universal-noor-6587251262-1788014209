# Byte-level starter tokenizer. Replace with a trained BPE/SentencePiece tokenizer for production.
def encode(text: str):
    return list(text.encode("utf-8", errors="ignore"))

def decode(ids):
    return bytes([int(i) % 256 for i in ids]).decode("utf-8", errors="ignore")
