from utils.utils import convert_categorial

import torch.nn as nn


class Loss():
    def __init__(self, gpu, num_cls1):
        #define criteria
        self.num_cls1 = num_cls1
        self.type_criterion = nn.CrossEntropyLoss()
        self.color_criterion = nn.BCELoss()

    def __call__(self, predictions, targets):
        """
        function to calculate loss when pass in parameters
        :param predictions: List of size 2 [torch.Tensor of type_preds, torch.Tensor of color_preds]
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
        type_targets = targets[:, :num_cls1]
        #get type_targets to CrossEntropyLoss target
        type_targets = convert_categorial(type_targets)

        color_targets = targets[:, num_cls1:]

        return type_targets, color_targets

