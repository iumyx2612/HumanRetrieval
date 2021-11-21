import argparse
import copy
from tqdm import tqdm
import time

from modeling.model import Model
from utils.loss import Loss
from utils.dataset_utils import *
from Classification_dict import dict as cls_dict
from utils.utils import get_imgsz, select_device
from utils.metrics import accuracy, AverageMeter

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset, train_loader, model, criterions, optimizer, epoch, device):
    losses = AverageMeter()
    type_losses = AverageMeter()
    color_losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = torch.tensor([0] * dataset.color_len)

    # switch to training mode
    model.train()

    start = time.time()
    for sample in tqdm(train_loader):
        # prepare targets
        inputs = (sample["image"] / 255.).to(device)
        type_targets = sample["type_onehot"]
        color_targets = sample["color_onehot"]
        targets = torch.cat([type_targets, color_targets], dim=1).to(device)

        # compute outputs
        outputs = model(inputs) # [type_pred, color_pred]
        total_loss, type_loss, color_loss = criterions(outputs, targets)

        # accuracy and loss
        type_acc, color_matching = accuracy(outputs, targets, dataset)
        acc1.update(type_acc.item(), inputs.size(0))
        acc2 += color_matching
        losses.update(total_loss.item(), inputs.size(0))
        type_losses.update(type_loss.item(), inputs.size(0))
        color_losses.update(color_loss.item(), inputs.size(0))

        # compute gradient and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # compute color acc
    num_color_dict = dataset.get_statistic()
    total_color = torch.tensor(list(num_color_dict.values()))
    color_acc = acc2 / total_color

    end = time.time()
    epoch_time = end - start
    print(f'Epoch: {epoch + 1} \t Time: {epoch_time}s')

    # logging
    s = ""
    for i in range(dataset.color_len):
        s += f'{dataset.clothes_color[i]} acc: {color_acc[i]} \t'
    print(f'Total loss: {losses.avg:.4f} \t'
          f'Type loss: {type_losses.avg:.4f} \t'
          f'Color loss: {color_losses.avg:.4f} \t'
          f'Acc: {acc1.avg:.4f} \n'
          f'{s}')
    return model, losses.avg # return metrics to save model


def val(dataset, val_loader, model, criterions, device):
    losses = AverageMeter()
    type_losses = AverageMeter()
    color_losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = torch.tensor([0] * dataset.color_len)

    model.eval()
    with torch.no_grad():
        for sample in tqdm(val_loader):
            inputs = (sample["image"] / 255.).to(device)
            type_targets = sample["type_onehot"]
            color_targets = sample["color_onehot"]
            targets = torch.cat([type_targets, color_targets], dim=1).to(device)

            # compute outputs
            outputs = model(inputs)  # [type_pred, color_pred]
            total_loss, type_loss, color_loss = criterions(outputs, targets)

            type_acc = accuracy(outputs, targets, dataset)
            acc1.update(type_acc.item(), inputs.size(0))
            losses.update(total_loss.item(), inputs.size(0))
            type_losses.update(type_loss.item(), inputs.size(0))
            color_losses.update(color_loss.item(), inputs.size(0))

        # compute color acc
        num_color_dict = dataset.get_statistic()
        total_color = torch.tensor(list(num_color_dict.values()))
        color_acc = acc2 / total_color
    s = ""
    for i in range(dataset.color_len):
        s += f'{dataset.clothes_color[i]} acc: {color_acc[i]} \t'
    print(f'Total loss: {losses.avg: .4f} \t Type loss: {type_losses.avg: .4f} \t Color loss: {color_losses.avg: .4f} \t'
          f'Type accuracy: {acc1.avg: .4f} \n {s}')


def run(args):
    # Init
    device = select_device(args.device)
    imgsz = get_imgsz(args.base_model)

    # Load data
    dataloader, dataset = create_dataloader(args.root_dir,
                                            args.csv_file,
                                            cls_dict,
                                            imgsz,
                                            args.batch_size,
                                            args.workers)

    # Build model
    if args.pretrained:
        model = Model(args.base_model,
                      True,
                      dataset.type_len,
                      dataset.color_len).to(device)
    elif args.resume:
        weights = torch.load(args.weights, map_location=device)
        model = Model(args.base_model,
                      False,
                      dataset.type_len,
                      dataset.color_len).to(device)
        model.load_state_dict(weights)
    else:
        model = Model(args.base_model,
                      False,
                      dataset.type_len,
                      dataset.color_len).to(device)

    # Build loss
    loss = Loss(device, dataset.type_len)

    # Build optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    if args.mode == "train":
        # Training loop
        print("Start training")
        print('-'*20)
        best_loss = 100
        for epoch in range(args.num_epochs):
            model, epoch_loss = train(dataset, dataloader, model, loss, optimizer, epoch, device)
            if epoch_loss < best_loss:
                print(f'Saving model with loss {epoch_loss}')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, args.save_dir)
    elif args.mode == "val":
        print("Start evaluation")
        print('-'*20)
        val(dataset, dataloader, model, loss, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train", type=str, choices=["train", "val"], help='train or validation mode')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--base_model', type=str, default='efficientnet-b0', help='base feature extractor')
    parser.add_argument('--pretrained', action='store_true', help='load EfficientNet with pretrained weights')
    parser.add_argument('--resume', action='store_true', help='load entire model with your weights')
    parser.add_argument('--weights', type=str, help='path to your weights to continue training')
    parser.add_argument('--root_dir', type=str, required=True, help='root directory stores all your image folder')
    parser.add_argument('--csv_file', type=str, required=True, help='name of your csv file contains info of your dataset')
    parser.add_argument('--save_dir', type=str, required=False, help='path to save your training model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
