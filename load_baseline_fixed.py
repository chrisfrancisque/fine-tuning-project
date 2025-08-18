import os
import torch
from collections.abc import Mapping

def _find_state_dict(obj):
    """
    Recursively find a dict that looks like a real state_dict
    (i.e., values are Tensors) inside possibly nested containers.
    """
    if isinstance(obj, Mapping):
        # If this mapping looks like a real state_dict, return it
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        # Common wrapper keys to check first
        for key in ("model_state_dict", "state_dict", "model", "weights"):
            if key in obj and isinstance(obj[key], Mapping):
                sd = _find_state_dict(obj[key])
                if sd is not None:
                    return sd
        # Fallback: search all nested mappings
        for v in obj.values():
            if isinstance(v, Mapping):
                sd = _find_state_dict(v)
                if sd is not None:
                    return sd
    return None

def _strip_prefixes(sd):
    fixed = {}
    for k, v in sd.items():
        nk = k
        # Handle common prefixes
        if nk.startswith("bert."):
            nk = nk[len("bert."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        fixed[nk] = v
    return fixed

def load_warmed_baseline(model, state_dict_path: str):
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Baseline state_dict not found: {state_dict_path}")

    raw = torch.load(state_dict_path, map_location="cpu")
    sd = _find_state_dict(raw)
    if sd is None:
        raise ValueError(
            "Could not locate a valid state_dict in the provided file. "
            "Expected keys like 'model_state_dict' or 'state_dict'."
        )

    sd = _strip_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Sanity: if we missed almost everything, warn loudly
    total_keys = sum(1 for _ in model.state_dict().keys())
    if len(missing) > 0.7 * total_keys:
        print(f"[baseline] WARNING: Loaded very few keys (missing={len(missing)} / {total_keys}). "
              "This suggests a mismatch between checkpoint and architecture.")

    # Verify embeddings shape
    we = model.bert.embeddings.word_embeddings.weight
    assert we.ndim == 2 and we.shape[0] in (30522, 28996), "Tokenizer/vocab mismatch?"

    if unexpected:
        print(f"[baseline] Unexpected (ignored) keys (first few): {sorted(list(unexpected))[:10]}")

    return model, missing, unexpected
