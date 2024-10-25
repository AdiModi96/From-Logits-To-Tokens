import numpy as np


def convert_logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def apply_repetition_penalty(logits: np.ndarray, frequencies: np.ndarray, repetition_penalty: float) -> np.ndarray:
    new_logits = logits.copy()
    new_logits[frequencies > 0] = new_logits[frequencies > 0] / repetition_penalty
    return new_logits


def apply_frequency_penalty(logits: np.ndarray, frequencies: np.ndarray, frequency_penalty: float) -> np.ndarray:
    return logits - (frequencies * frequency_penalty)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / temperature


def select_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    new_logits = logits.copy()
    new_logits[np.argpartition(new_logits, -k)[:-k]] = np.nan
    return new_logits


def select_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    assert p >= 0 and p <= 1, 'The value of cumulative probability should be between 0 & 1'

    new_logits = logits.copy()

    probabilities = convert_logits_to_probabilities(logits)
    sorted_indices = np.argsort(1 - probabilities)

    if probabilities[0] > p:
        idx = 0
    else:
        for idx in range(1, len(sorted_indices) + 1):
            if probabilities[:idx].sum() > p:
                break

    new_logits[sorted_indices[idx:]] = np.nan

    return new_logits