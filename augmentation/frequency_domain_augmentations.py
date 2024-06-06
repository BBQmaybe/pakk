import torch
import numpy as np

def mix_up(melspec_a, label_a, melspec_b, label_b, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    augmented_melspecs = melspec_a * lam + melspec_b * (1 - lam)
    augmented_labels = label_a * lam + label_b * (1 - lam)

    return augmented_melspecs, augmented_labels