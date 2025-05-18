#Authors: Timo Michoud, Sina Röllin, Veronika Podliesnova


#Import necessary libraries
import numpy as np


def custom_f1_score(y_true, y_pred):
    """Compute image-wise average F1 score based on counts""" 
    eps = 1e-7
    f1_scores = []
    for true, pred in zip(y_true, y_pred):
        TP = np.sum(np.minimum(true, pred))
        FP_N = np.sum(np.abs(true - pred))
        if TP + FP_N == 0:
            f1 = 1.0  # perfect zero prediction
        else:
            f1 = (2 * TP) / (2 * TP + FP_N + eps)
        f1_scores.append(f1)
    return np.mean(f1_scores)