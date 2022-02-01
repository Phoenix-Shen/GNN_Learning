import torch as t
import numpy as np


def calculate_accuracy(output: t.Tensor, labels: t.Tensor) -> float:
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)


def set_seed(seed: int, cuda: bool) -> None:
    t.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        t.cuda.manual_seed(seed)
