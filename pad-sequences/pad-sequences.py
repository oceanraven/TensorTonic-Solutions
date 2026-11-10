import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = 0
        for seq in seqs:
            max_len = max_len if max_len > len(seq) else len(seq)

    res = np.full((len(seqs), max_len), pad_value)

    for i, seq in enumerate(seqs):
        res[i, :min(len(seq), max_len)] = seq[:min(len(seq), max_len)]
    return res