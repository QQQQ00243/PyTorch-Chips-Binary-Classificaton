import argparse
import torchvision.transforms as T
import os

from PIL import Image
from loguru import logger
from utils import get_dataloader

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def make_args():
    parser = argparse.ArgumentParser(description="Show chip data")
    parser.add_argument("--train-root", type=str, default="./data/Chips/train/",
                        help="root to training datasets")
    parser.add_argument("--test-root", type=str, default="./data/Chips/test/",
                        help="root to training datasets")
    parser.add_argument("--csv-file", type=str, default="./data/Chips/answer.csv",
                        help="idx to class csv file of test dataset")
    return parser


class ChipsTrainDataset(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
    ):
        super(ChipsTrainDataset, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class ChipsTestDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root,
        transform=None,
    ):
        super(ChipsTestDataset, self).__init__()
        self.idx_to_classes = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_classes)
    
    def __getitem__(self, idx):
        img_file = os.path.join(
            self.root,
            str(idx) + ".png",
        )
        img = Image.open(img_file)
        if self.transform is not None:
            img = self.transform(img)
        label = self.idx_to_classes.iloc[idx, 0]
        return img, label


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        h, w = img.size
        if h > w:
            img = img.rotate(angle=90, expand=2)
        img = img.resize(self.output_size)
        return img


def get_ChipsDataloader(
    train_root,
    test_root,
    csv_file,
    batch_size,
    test_batch_size,
    valid_split,
    num_workers=4,
):
    transforms = [
        Rescale((180, 250)),
        T.RandomHorizontalFlip(),
        # T.RandomCrop(size=200, padding=2),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    logger.info(f"Augmentation plan: {transforms}\n")
    test_transforms = [
        Rescale((180, 250)),
        # T.Resize((250, 250)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    transform = T.Compose(transforms)
    test_transform = T.Compose(test_transforms)
    
    train_dataset = ChipsTrainDataset(root=train_root, transform=transform)
    logger.info(f"class to idx: {train_dataset.class_to_idx}\n")

    valid_dataset = ChipsTrainDataset(root=train_root, transform=test_transform)
    test_dataset = ChipsTestDataset(
        root=test_root,
        csv_file=csv_file,
        transform=test_transform,
    )

    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        num_workers=num_workers,
        valid_split=valid_split,
    )
    return train_loader, val_loader, test_loader


def plot_dist(bins):
    '''
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = ChipsTrainDataset(
        root=args.train_root,
        img_size=args.img_size,
        transform=transform,
    )
    '''
    widths, heights = [], []
    list_path = glob.glob("./data/Chips/train/*/*")
    for path in list_path:
        img = Image.open(path)
        
        widths.append(max(img.size))
        heights.append(min(img.size))
    '''
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        if img.size[0] < img.size[1]:
            img = img.rotate(90, Image.NEAREST, expand = 1)
        widths.append(img.size[0])
        heights.append(img.size[1])
    '''

    plt.hist(x=widths, bins=bins)
    plt.hist(x=heights, bins=bins)
    plt.show()

    print(np.mean(widths))
    print(np.mean(heights))


def main():
    '''
    test_dataset = ChipsTestDataset(
        csv_file="./data/Chips/answer.csv",
        root_dir="./data/Chips//",
        img_size=(150, 75),
        transform=transform,
    )
    for i in range(len(test_dataset)):
        print(test_dataset[i])
    '''
    
    args = make_args().parse_args()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = ChipsTrainDataset(
        root=args.train_root,
        img_size=args.img_size,
        transform=transform,
    )
    print(train_dataset.class_to_idx)
    
    '''
    args = make_args().parse_args()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = ChipsTestDataset(
        csv_file=args.csv_file,
        root=args.test_root,
        img_size=args.img_size,
        transform=transform,
    )
    
    for i in range(len(test_dataset)):
        print(test_dataset[i][1])
    '''


if __name__ == "__main__":
    main()
