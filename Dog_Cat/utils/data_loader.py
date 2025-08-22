from torchvision import transforms, datasets
from torch.utils.data import random_split
import os


def get_train_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform


def get_test_transforms():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return test_transform


def load_datasets(data_root, train_ratio=0.8):

    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    full_train = datasets.ImageFolder(os.path.join(data_root, 'train'), train_transform)
    test_data = datasets.ImageFolder(os.path.join(data_root, 'test'), test_transform)

    train_size = int(len(full_train) * train_ratio)
    val_size = len(full_train) - train_size
    train_data, val_data = random_split(full_train, [train_size, val_size])

    return train_data, val_data, test_data, full_train.classes