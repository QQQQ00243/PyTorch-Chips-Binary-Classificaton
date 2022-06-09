import os
import torch
import argparse

import pandas as pd
import torch.nn as nn

from loguru import logger
from Task3.models import M
from Task3.load_data import get_ChipsDataloader


def make_args():
    parser = argparse.ArgumentParser(description="Chips--Train")
    # --------------------configurations of dataset -------------------------------
    parser.add_argument("--valid-split", default=0.2,
                        help="Split ratio of training dataset")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N",
                        help="input batch size for testing")
    parser.add_argument("--img-size", type=int, nargs="+", default=[280, 150],
                        help="Resize img to a given size.")

    parser.add_argument("--ImageAugmentation", action="store_true",
                        help="Enable image augmentation")

    # ---------------------configurations of saving------------------------------
    parser.add_argument("--ckpt-file", type=str, 
                        help="checkpoints file")
    parser.add_argument("--imgs-dir", type=str, default="./imgs",
                        help="directory to save images")
    parser.add_argument("--logs-dir", type=str, default="./logs",
                        help="directory to save log file")
    parser.add_argument("--train-root", type=str, default="./data/Chips/train/",
                        help="root to training datasets")
    parser.add_argument("--test-root", type=str, default="./data/Chips/test/",
                        help="root to training datasets")
    parser.add_argument("--csv-file", type=str, default="./data/Chips/answer.csv",
                        help="idx to class csv file of test dataset")
    parser.add_argument("--res-dir", type=str,
                        help="directory to save results")                    

    # ---------------------configurations of backbone ------------------------------
    parser.add_argument("--backbone-name", type=str, default=None,
                        choices=["vgg", "resnet"], help="backbone name")
    return parser


def get_device(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Using {}\n", device)
    return device


def make_dir(args):
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)
    if not os.path.exists(args.logs_dir):
        os.mkdir(args.logs_dir)


def pred(model: nn.Module, dataloader, crit):
    preds = []
    model.eval()
    device = get_device()
    model.to(device)
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += crit(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            pred_npy = pred.view_as(target).numpy()
            for i in pred_npy:
                preds.append(i)
            acc += pred.eq(target.view_as(pred)).sum().item() / len(data)
    loss /= len(dataloader)
    acc /= len(dataloader)
    return loss, acc, preds


def main():
    args = make_args().parse_args()
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}", arg, value)
    make_dir(args)
    num_classes = 2
    model = M(
        backbone_name=args.backbone_name,
        num_classes=num_classes,
    )
    model.load_state_dict(
        torch.load(
            args.ckpt_file,
            map_location=torch.device('cpu')
        )
    )
    logger.info("model:\n {}", model)
    _, val_loader, test_loader = get_ChipsDataloader(args)
    crit = nn.CrossEntropyLoss()
    loss, acc, preds = pred(
        model=model,
        dataloader=val_loader,
        crit=crit,
    )
    logger.info(f"loss: {loss}\tacc: {acc}\n") 

    files = [f"{i}.png" for i in range(len(preds))]
    res = {"files": files, "preds": preds}
    res = pd.DataFrame(res)
    res_file = os.path.join(args.res_dir, "res.csv")
    res.to_csv(res_file, index=False)
    logger.info(f"Saving result to {res_file}\n")
    logger.info("Finish!\n")


if __name__ == "__main__":
    main()



