from torch import Tensor


def accuracy(logits: Tensor, y: Tensor) -> float:
    """Fraction of correct predictions; logits (B, C), y (B,)."""
    pred=logits.argmax(dim=1)
    return float((pred==y).float().mean().item())