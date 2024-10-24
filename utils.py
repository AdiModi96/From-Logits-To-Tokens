import numpy as np


def apply_repetition_penalty(logits: np.ndarray, frequency: np.ndarray, repetition_penalty: float) -> np.ndarray:
    repetition_penalty_adjusted_logits = logits.copy()
    repetition_penalty_adjusted_logits[frequency > 0] = repetition_penalty_adjusted_logits[frequency > 0] / repetition_penalty
    return repetition_penalty_adjusted_logits
