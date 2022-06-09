## Task 3

**Description**: Classify normal and abnormal chips

**Show data**

```bash
python3 -m Task3.showchips --show
```

![](./assets/imgs/Chips_Examples.svg)



## CNN (Backbone-ResNet18)

### Resize data to mean value of size (180, 250) Test Accuracy: 0.894

<img src="./assets/imgs/resize.svg" style="zoom:50%;" />

### Random Horizontal Flip Test Accuracy: 0.906

<img src="./assets/imgs/RandomHorizontalFlip.svg" style="zoom:50%;" />

### Random Horizontal and Vertical Flip Test Accuracy: 0.906

<img src="./assets/imgs/RandomHVFlip.svg" style="zoom:50%;" />

### Random Horizontal Flip and Random Crop Test Accuracy: 0.879

<img src="./assets/imgs/HF_RandomCrop.svg" style="zoom:50%;" />

### Auto Augmentation with CIFAR10 Policy Test Accuracy: 0.824

<img src="./assets/imgs/AutoAugmentCIFAR10.svg" style="zoom:50%;" />

### Random Erasing + Random Horizontal Flip: Test Accuracy: 0.906

<img src="./assets/imgs/RandomEraseHP.svg" style="zoom:50%;" />



## Siamese Network









* no image augmentation - Test acc: 92.2

```bash
python3 -m Task3.train --backbone-name "resnet" --epochs 30 --milestone1 10 --milestone2 20
```

* Horizontal flip - Test acc: 91.6

```bash
python3 -m Task3.train --backbone-name "resnet" --epochs 50 --milestone1 25 --milestone2 35 --ImageAugmentation
```

* Vertical flip - Test acc: 92.8

```bash
python3 -m Task3.train --backbone-name "resnet" --epochs 50 --milestone1 25 --milestone2 35 --ImageAugmentation
```

* Center Crop - Test acc:

```bash
python3 -m Task3.train --backbone-name "resnet" --epochs 50 --milestone1 25 --milestone2 35 --ImageAugmentation
```
