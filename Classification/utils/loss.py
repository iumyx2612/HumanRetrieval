from utils.utils import convert_categorial

import torch.nn as nn
import torch


class CrossEntropyLabelSmoothing(nn.Module):
    """
    This loss class works like CrossEntropyLoss from Pytorch with smoothing factor
    It accepts target as a number and pred is logits
    Label smoothing from Inception-v2 paper
    Explain:
        label' = (1-smoothing) * label + smoothing/num_labels
    """
    def __init__(self, smoothing=0.1, dim=-1):
        super(CrossEntropyLabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim) # apply log(softmax(x)) on the logits
        with torch.no_grad():
            true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Loss():
    def __init__(self, num_cls1, label_smoothing0=None, label_smoothing1=None):
        #define criteria
        self.num_cls1 = num_cls1
        if label_smoothing0 is not None:
            self.type_criterion = CrossEntropyLabelSmoothing()
        else:
            self.type_criterion = nn.CrossEntropyLoss()
        self.color_criterion = nn.BCELoss()

    def __call__(self, predictions, targets):
        """
        function to calculate loss when pass in parameters
        :param predictions: List of size 2 [type_preds: torch.Tensor, color_preds:torch.Tensor]
        :param targets: Tensor of size (batch, type_target + color_target)
        :return: total loss
        """
        type_preds = predictions[0]
        color_preds = predictions[1]
        type_targets, color_targets = self.build_targets(targets, num_cls1=self.num_cls1)
        type_loss = self.type_criterion(type_preds, type_targets)
        color_loss = self.color_criterion(color_preds, color_targets)
        total_loss = type_loss + color_loss
        return total_loss, type_loss, color_loss

    def build_targets(self, targets, num_cls1):
        """
        build suitable targets for criterions
        :param predictions: torch.Tensor of size (batch, type_pred + color_pred)
        :param targets: torch.Tensor of size (batch, type_target + color_target)
        :param num_cls1: number of types
        """
        type_targets = targets[:, :num_cls1] # one hot vector
        #get type_targets to CrossEntropyLoss target
        type_targets = convert_categorial(type_targets)

        color_targets = targets[:, num_cls1:]

        return type_targets, color_targets


def test_label_smoothing():
    import tensorflow as tf
    import tensorflow.keras as keras

    # TF implementation of Label Smoothing Cross Entropy
    y_true = tf.Variable([
        [0, 1, 0]
    ], dtype=tf.float32)
    y_pred = tf.Variable([
        [1, 8, 1]
    ], dtype=tf.float32)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    result = loss(y_true, y_pred)
    print(keras.backend.eval(result))

    # My implementation based on Pytorch
    y_true = torch.tensor([1])
    y_pred = torch.tensor([[1, 8, 1]], dtype=torch.float)
    loss = CrossEntropyLabelSmoothing(smoothing=0.1)
    result = loss(y_pred, y_true)
    print(result.numpy())


if __name__ == '__main__':
    test_label_smoothing()
    pass



