import os
import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from loguru import logger
from utils import train, validate, make_dir, adjust_lr
from Task3.load_data import get_ChipsDataloader


def make_args():
    parser = argparse.ArgumentParser(description="Chips--Train")
    # --------------------configurations of dataset -------------------------------
    parser.add_argument("--valid-split", default=0.1, type=float,
                        help="Split ratio of training dataset")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N",
                        help="input batch size for testing")

    # ---------------------configurations of training------------------------------
    parser.add_argument("--epochs", type=int, default=50, metavar="N",
                        help="number of epochs to train")
    parser.add_argument("--milestone1", type=int, default=15, metavar="N",
                        help="number of epochs to train")
    parser.add_argument("--milestone2", type=int, default=25, metavar="N",
                        help="number of epochs to train")

    # ---------------------configurations of learning rate scheduler -----------------
    parser.add_argument("--init-lr", type=float, default=0.1,
                        help="initial learning rate")

    # ---------------------configurations of saving------------------------------
    parser.add_argument("--ckpts-dir", type=str, default="./ckpts",
                        help="directory to save checkpoints")
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

    return parser


def get_device(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Using {}\n", device)
    return device


def get_prefix()->str:
    return "ResNet18"


def fit(
    model: nn.Module,
    crit,
    epochs,
    init_lr,
    ckpt_file,
    train_loader,
    val_loader,
    milestone1,
    milestone2,
):
    device = get_device()
    model = model.to(device)
    train_loss, train_acc, val_loss, val_acc = [[] for _ in range(4)]
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, epochs+1):
        # training
        train_loss_, train_acc_ = train(
            model=model,
            device=device,
            criterion=crit,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        train_loss.append(train_loss_)
        train_acc.append(train_acc_)

        # validation
        val_loss_, val_acc_ = validate(
            model=model,
            device=device,
            criterion=crit,
            val_loader=val_loader,
        )
        val_loss.append(val_loss_)
        val_acc.append(val_acc_)

        logger.info(f"Train Epoch: {epoch} / {epochs} LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Train Loss: {train_loss_:.5f}\tTrain Accuracy: {train_acc_:.5f}")
        logger.info(f"Valid Loss: {val_loss_:.5f}\tValid Accuracy: {val_acc_:.5f}\n")
        adjust_lr(
            optimizer=optimizer,
            epoch=epoch,
            milestone1=milestone1,
            milestone2=milestone2,
        )

    logger.info(f"Saving model to {ckpt_file}\n")
    torch.save(model.state_dict(), ckpt_file)
    history = {
        "train_history": {"train_accuracy": train_acc, "train_loss": train_loss},
        "val_history": {"val_accuracy": val_acc, "val_loss": val_loss},
    }
    return history


def main():
    args = make_args().parse_args()
    make_dir(args)
    prefix = get_prefix()
    log_file = os.path.join(args.logs_dir, prefix+"_{time}.log")
    logger.add(log_file)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}", arg, value)

    num_classes = 2
    model = models.resnet18(num_classes=num_classes, pretrained=False)
    logger.info("model:\n {}", model)

    ckpt_file = os.path.join(args.ckpts_dir, prefix+".pth")
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = get_ChipsDataloader(
        train_root=args.train_root,
        test_root=args.test_root,
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        valid_split=args.valid_split,
    )
    history = fit(
        model=model,
        crit=crit,
        epochs=args.epochs,
        init_lr=args.init_lr,
        ckpt_file=ckpt_file,
        train_loader=train_loader,
        val_loader=val_loader,
        milestone1=args.milestone1,
        milestone2=args.milestone2,
    )
    
    test_loss, test_acc = validate(
        model=model,
        device=get_device(),
        criterion=crit,
        val_loader=test_loader,
    )
    logger.info(f"Test Loss: {test_loss:.5f}\tTest Accuracy: {test_acc:.5f}")
    history["test"] = {"test_loss": test_loss, "test_acc": test_acc}
    logger.info("History:{}\n", history)
    
    logger.info("Saving log to {}", log_file)
    logger.info("Finish!")


if __name__ == "__main__":
    main()
