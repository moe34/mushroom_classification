import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
#load model
#load dataset 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5)),((0.5,0.5,0.5))
])
test_dataset = data.DataLoader(root="data/test", train=False, transform=transform)
test_loader = data.DataLoader(test_dataset,batch_size=4,shuffle=False)

mushroom_classes = ["hiratake", "tsukiyotake"]
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        #data(1*2のベクトルをtest枚数分含んだテンソルの、ベクトルごとの最大値を返す(1=row,0=col))
        #torch.max()は最大値と１を返す　最大値は不要なので_（＝適当な名前の変数）で受け取っている
        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy : ", correct/total)




