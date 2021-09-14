import os
from PIL import Image
import glob

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt

def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):      
        if os.path.isfile(p): #isfileで存在確認
            os.remove(p)

def make_filepath_list(dataset_dir):
    train_file_list = []
    valid_file_list = []

    for dir in os.listdir(dataset_dir):
        file_dir = os.path.join(dataset_dir, dir)
        file_list = os.listdir(file_dir)

        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join(dataset_dir, dir, file) for file in file_list[:num_split]]
        valid_file_list += [os.path.join(dataset_dir, dir, file) for file in file_list[num_split:]]
    return train_file_list, valid_file_list

class ImageTransform(object):
    def __init__(self, resize=256, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
        self.data_transform = {
            "train" : transforms.Compose([
                transforms.RandomVerticalFlip(), #flip upside down
                transforms.RandomHorizontalFlip(), #flip horizontal
                transforms.Resize((resize, resize)), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std) #normalize of color channel
            ]),
            "valid" : transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase = "train"):
        return self.data_transform[phase](img)

class mushroomDataset(data.Dataset):
    def __init__(self, file_list, classes, transform=None, phase= "train"):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        label = self.file_list[index].split('/')[1]
        label = self.classes.index(label) #convert label name to number

        return img_transformed, label

class  Net(nn.Module):
    def __init__(self):
        #nn.Moduleの初期化関数を起動
        super(Net, self).__init__()
        #self.xxxで各変数を定義
        self.relu = nn.ReLU()
        #celi_mode=Trueにすると出力サイズを切り上げる
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,ceil_mode=False)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size)

        self.



if __name__ == "__main__":
    remove_glob("dataset/.DS_Store")
    remove_glob("dataset/*/.DS_Store")
    train_file_list, valid_file_list = make_filepath_list("dataset/")
    print("train num:", len(train_file_list))
    print("valid num:", len(valid_file_list))
    transform = ImageTransform()
    mushroom_classes = ["hiratake", "tsukiyotake"]
    train_dataset = mushroomDataset(file_list=train_file_list, classes=mushroom_classes,transform=transform, phase="train")
    valid_dataset = mushroomDataset(file_list=valid_file_list, classes=mushroom_classes,transform=transform, phase="valid")
    index = 0
    print(train_file_list[0])
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])
    train_dataloader = data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size = 2, shuffle=False)
    dataloaders_dict = {
        "train" : train_dataloader,
        "valid" : valid_dataloader
    }
    batch_iterator = iter(dataloaders_dict["train"])
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)
