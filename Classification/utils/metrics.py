import torch
from utils.utils import convert_categorial
from utils.datasets import ClothesClassificationDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = torch.tensor([0], dtype=torch.float)
        self.avg = torch.tensor([0], dtype=torch.float)
        self.sum = torch.tensor([0], dtype=torch.float)
        self.count = torch.tensor([0], dtype=torch.int)

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, dataset: ClothesClassificationDataset):
    """Computes the accuracy for clothes type"""
    with torch.no_grad():
        batch_size = target.size(0)

        # split target
        type_target = target[:, :dataset.type_len] # torch.Tensor(batch, dataset.type_len)
        type_target = convert_categorial(type_target) # torch.Tensor(batch)
        color_target = target[:, dataset.type_len:] # torch.Tensor(batch, dataset.color_len)

        # convert output
        type_output = output[0] # torch.Tensor(batch, dataset.type_len)
        type_output = torch.softmax(type_output, dim=1)
        type_output = torch.argmax(type_output, dim=1) # torch.Tensor(batch)
        color_output = output[1]
        color_output = (color_output > 0.5).type(torch.int) # torch.Tensor(batch, dataset.color_len)

        # compute acc for type
        type_correct = type_target.eq(type_output).type(torch.int)
        type_acc = type_correct.sum().mul_(100) / batch_size # torch.Tensor

        # compute acc for color
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
        confusion_vector = color_output / color_target
        color_matching = torch.tensor([0] * dataset.color_len)
        for i in range(dataset.color_len):
            color_matching[i] = torch.sum(confusion_vector[:, i] == 1) # torch.Tensor(color_len)
    return type_acc, color_matching # torch.Tensor(1), torch.Tensor(color_len)


def fitness(metrics, weights=None, useloss=True):
    """ Model fitness as weighted combination of metrics
    weight for total_loss, type_loss, color_loss, type_acc, avg_color_acc

    """
    assert metrics.shape == weights.shape, "Metrics and weights combination must have same shape"
    if weights is None:
        weights = torch.tensor([0.5, 0.2, 0.3, 0.0, 0.0]) # default use loss not acc
        # the smaller the loss the better
        return 1.0 - torch.matmul(metrics, weights) # -> the
    elif not useloss:
        if weights is None:
            weights = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5])



if __name__ == '__main__':
    outputs = [torch.rand((4, 4)), 0]
    targets = torch.tensor([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    res = accuracy(outputs, targets, 4)
    print(res)