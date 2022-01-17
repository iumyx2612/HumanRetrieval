import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

import tensorflow as tf
import tensorflow.keras as keras


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        #lsm = F.log_softmax(inputs, -1)
        lsm = inputs

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == '__main__':
    # TF
    y_true = tf.Variable([
        [0, 1, 0]
    ], dtype=tf.float32)
    y_pred = tf.Variable([
        [1, 8, 1]
    ], dtype=tf.float32)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=True)
    result = loss(y_true, y_pred)
    print(keras.backend.eval(result))

    # Pytorch
    y_pred = torch.tensor([[1, 8, 1]], dtype=torch.float)
    y_true = torch.tensor([1])
    #print(F.log_softmax(y_pred).float())
    loss = LabelSmoothingLossCanonical(smoothing=0.1)
    result = loss(y_pred, y_true)
    print(result.numpy())

    # test
    a = torch.tensor([1, 8, 1], dtype=torch.float)
    b = F.softmax(a, dim=0) * 100
    print(b)
