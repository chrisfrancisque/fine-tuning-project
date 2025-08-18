import os
import torch

def load_warmed_baseline(model, state_dict_path: str):
    """
    Loads a warmed baseline state_dict and fixes 'bert.'-prefixed keys if present.
    Works with files saved as plain state_dict (not HF save_pretrained).
    """
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Baseline state_dict not found: {state_dict_path}")

    sd = torch.load(state_dict_path, map_location="cpu")
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    fixed = {}
    for k, v in sd.items():
        nk = k[5:] if k.startswith("bert.") else k
        fixed[nk] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)

    # Sanity checks
    if unexpected:
        print(f"[baseline] Unexpected keys ignored: {sorted(list(unexpected))[:10]} ...")
    # Verify embeddings shape & classifier presence (may be randomly init)
    we = model.bert.embeddings.word_embeddings.weight
    assert we.shape[0] in (30522, 28996) and we.ndim == 2, "Tokenizer/vocab mismatch?"

    return model, missing, unexpected
