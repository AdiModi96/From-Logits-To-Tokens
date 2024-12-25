import numpy as np


def convert_logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits)
    return exp_logits / np.nansum(exp_logits)


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

    pmf = convert_logits_to_probabilities(logits)
    sorted_idxes = np.argsort(pmf)[::-1]
    sorted_idxes_inv = np.argsort(sorted_idxes)

    sorted_pmf = pmf[sorted_idxes]
    sorted_logits = logits[sorted_idxes]
    cumulative_sorted_pmf = np.cumsum(sorted_pmf)

    sorted_logits[cumulative_sorted_pmf > p] = np.nan
    return sorted_logits[sorted_idxes_inv]