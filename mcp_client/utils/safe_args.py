def _safe_args(d: dict, limit=200):
    out = {}
    for k, v in d.items():
        if k.endswith("_text") and isinstance(v, str):
            out[k] = f"<{len(v)} chars>"
        else:
            out[k] = v
    return out