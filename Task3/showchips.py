import os
import argparse

import matplotlib.pyplot as plt
import torchvision.transforms as T

from loguru import logger
from utils import get_idx_to_class
from .load_data import ChipsTrainDataset, Rescale
from tools.plot import show_augment_tensor


def make_args():
    parser = argparse.ArgumentParser(description="Show Chips")
    parser.add_argument("--train-root", default="./data/Chips/train/", 
                        type=str, help="root of dataset")
    parser.add_argument("--imgs-dir", default="./imgs",
                        type=str, help="directory to images")
    parser.add_argument("--show", action="store_true",
                        help="show CIFAR10 examples")
    parser.add_argument("--show-augment", action="store_true",
                        help="show augmentation") 
    return parser


def make_dir(args):
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)


def show_Chips(
    root,
    title,
    img_file,
    num_per_row,
    num_instances,
    title_size=25,
    subtitle_size=15,
):
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [10, 10]
    nrows = -(-num_instances // num_per_row)
    ncols = num_per_row
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    fig.set_size_inches(10, 10)
    dataset = ChipsTrainDataset(root=root)
    idx_to_class = get_idx_to_class(dataset.class_to_idx)
    for i in range(num_instances):
        img, label = dataset[i]
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        class_ = idx_to_class[label]
        plt.title('{}'.format(class_), fontsize=subtitle_size)
        plt.imshow(img)
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title, fontsize=title_size)
    plt.savefig(img_file)
    plt.show()


def get_augmenter():
    logger.info("Using image augmentation.")

    # augmenter = [T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)]
    
    # augmenter = [T.RandomPerspective(distortion_scale=0.4, p=0.4)]

    augmenter = [
        Rescale((180, 250)),
        # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        # T.RandomCrop(size=180, padding=2),
        T.ToTensor(),
        T.RandomErasing(),
    ]

    logger.info(f"{augmenter}\n")
    return T.Compose(augmenter)


def main():
    args = make_args().parse_args()
    dataset = ChipsTrainDataset(root=args.train_root)    

    if args.show:
        img_file=os.path.join(args.imgs_dir, "Chips_Examples.svg"),
        logger.info(f"Saving Chips examples to {img_file}.\n")
        show_Chips(
            root=args.train_root,
            title="Chips Examples",
            img_file=os.path.join(args.imgs_dir, "Chips_Examples.svg"),
            num_per_row=8,
            num_instances=64,
        )
    if args.show_augment:
        img_file = os.path.join(args.imgs_dir, "augment.svg")
        logger.info(f"Saving image to {img_file}.\n")
        augmenter = get_augmenter()
        show_augment_tensor(
            file=img_file,
            dataset=dataset,
            augmenter=augmenter,
            num_augments=5,
            num_instances=6,
        )


if __name__ == "__main__":
    main()