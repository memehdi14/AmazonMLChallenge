"""Fine-tuned multilingual cross-encoder (xlm-roberta-base / mdeberta-v3-base, MIT)."""


def build_pairs_text(pairs, s1, s23):
    """Format each pair as '[name] | [address]' text for both records."""
    raise NotImplementedError


def train_ce(pairs_text, labels, cfg):
    """Fine-tune the cross-encoder as a binary classifier; save a checkpoint."""
    raise NotImplementedError


def predict_ce(pairs, ckpt):
    """Return a match probability per pair from a saved checkpoint."""
    raise NotImplementedError
