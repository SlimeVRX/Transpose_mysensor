import sys
import os
from utils import compute_metrics, rotation_6d_to_matrix, smpl_reduced_to_full, smpl_reduced_to_full_torch
sys.path.append(os.getcwd())
import torch.nn.functional as F
import torch
import numpy as np
def compute_angle_dif(prediction, target):
    """
    prediction (batch_size, seq_len, 15, 6)
    """
    b, seq, dim = prediction.shape
    prediction = prediction.reshape(b, seq, 15, 6)
    target = target.reshape(b, seq, 15, 6)
    
    prediction = rotation_6d_to_matrix(prediction)
    target = rotation_6d_to_matrix(target)

    prediction = smpl_reduced_to_full(prediction.reshape(b * seq, 15 * 9))
    target = smpl_reduced_to_full(target.reshape(b * seq, 15 * 9))

    seq_angel_error, _ = compute_metrics(prediction, target)
    seq_angel_error = seq_angel_error.reshape(b, seq, -1)
    seq_angel_error = np.sum(seq_angel_error, axis=(1, 2))
    return np.mean(seq_angel_error)

def poseLoss(output, target):
    c = output - target
    res = c * c
    return res.sum(dim=(1, 2)).mean()

def crossEntropy(output, target):
    term = - target * torch.log(output) - (1 - target) * torch.log(1 - output)
    term = term.sum(dim=(1, 2)).mean()
    return term

def ver_loss(output, target, n):
    s = torch.zeros(1).cuda()
    result = output - target
    result_norm = torch.norm(result, p=2, dim=-1)
    if n == 1:
        return result_norm.sum(dim=1)
    arr = torch.split(result_norm, n)
    for item in arr:
        s += item.sum()
    return s / len(arr) if len(arr) != 0 else s


def ver_n_loss(output, target):
    result = ver_loss(output, target, 1) + ver_loss(output, target, 3) + ver_loss(
        output, target, 9) + ver_loss(output, target, 27)
    return result.mean()


def foot_accuracy(output, target):
    output = (output > 0.5).int()
    target = target.int()
    _acc = ((output == target).sum(dim=-1) == 2).float()
    return _acc.mean() * 100

if __name__ == "__main__":
    output = torch.randn((256, 300, 2))
    target = torch.randn((256, 300, 2))

    out = foot_accuracy(output, target)
    print(out)
