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
import numpy as np

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
        img = Image.open(img_path).convert("RGB")
        
        img_transformed = self.transform(img, self.phase)
        label = self.file_list[index].split('/')[1]
        label = self.classes.index(label) #convert label name to number

        return img_transformed, label
#conv1 relu(活性化) pooling（画像サイズの縮小) conv2 fully connect
class  Net(nn.Module):
    def __init__(self):
        #nn.Moduleの初期化関数を起動
        super(Net, self).__init__()
        #self.xxxで各変数を定義
        self.relu = nn.ReLU()
        #celi_mode=Trueにすると出力サイズを切り上げる
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,ceil_mode=False)
        
        #convolutional network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        #fully connected layers
        #fc1(input_features, hidden)
        #fc2(hidden, output_features)
        self.fc1 = nn.Linear(in_features=64*126*126, out_features=64)
        self.fc2 = nn.Linear(64,2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print("x.shape=",x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #分類問題なので活性化関数としてsoftmax関数を使用
        x = F.softmax(x,dim=1)

        return x





if __name__ == "__main__":
    toch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
    train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size = 4, shuffle=False)
    dataloaders_dict = {
        "train" : train_dataloader,
        "valid" : valid_dataloader
    }
    batch_iterator = iter(dataloaders_dict["train"])
    inputs, labels = next(batch_iterator)
    print(inputs.size())#(torch.Size([batch_size, channel,height,width]))
    print(labels)
    net = Net()
    print(net)
    #損失関数の定義
    criterion = nn.CrossEntropyLoss()
    #最適化手法の定義
    optimizer=optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 30

    for epoch in range(num_epochs):
        print(("Epoch {}/{}").format(epoch+1, num_epochs))
        print("-------------")
        
        for phase in ["train","valid"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            #epochごとの損失和
            epoch_loss = 0.0
            #正解数
            epoch_corrects = 0
            best_acc = 0.0

            epoch_loss_list = []
            epoch_accuracy_list = []

            for inputs, labels in dataloaders_dict[phase]:
                #optimizerの初期化
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, axis=1)
                    
                    if phase == "train":
                        #逆伝播の計算
                        loss.backward()
                        #パラメーターの更新
                        optimizer.step()
                
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)
            epoch_loss = epoch_loss /len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            epoch_loss_list.append(epoch_loss)
            epoch_accuracy_list.append(epoch_acc)
            if phase =="train":
                if epoch_acc > best_acc:
                    torch.save(net.state_dict(), "best.pth")
                    best_acc = epoch_acc
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
        
    fig, ax = plt.subplots()  
    ax.set_xlabel("Epoch")
    epoch_list = np.linspace(1,num_epochs, 1)
    ax.plot(epoch_list, epoch_loss_list, color="blue", label="loss")
    ax.plot(epoch_list, epoch_accuracy_list, color="orange", label="accuracy")
    ax.legend()
    plt.savefig("result.png")
    plt.show()


