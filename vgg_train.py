import torch
from torch import nn
import numpy as np
from torch.optim import lr_scheduler
import os
from tqdm import tqdm

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.utils.data as Data

import matplotlib.pyplot as plt

vgg16 = models.vgg16(pretrained=True)

for param in vgg16.parameters():
    param.requires_grad = False

vgg16.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 49)
)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 224
    transforms.RandomHorizontalFlip(),  # 默认概率0.5
    transforms.ToTensor(),
    # 图像标准化处理
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # 图像标准化处理
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data_dir = r'train'
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
# ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
train_data_loader = Data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_data_dir = r'val'
val_data = ImageFolder(val_data_dir, transform=val_data_transforms)
val_data_loader = Data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0)

# print("训练集样本：", len(train_data.targets))
# print("测试集样本：", len(val_data.targets))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = vgg16
model.cuda()

# 定义一个损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(tqdm(dataloader)):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print('train_loss' + str(train_loss))
    print('train_acc' + str(train_acc))
    return train_loss, train_acc


# 定义一个验证函数
def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss' + str(val_loss))
    print('val_acc' + str(val_acc))
    return val_loss, val_acc


# 定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 50
min_acc = 0

if __name__ == '__main__':
    for t in range(epoch):
        # lr_scheduler.step()
        print(f"epoch{t + 1}\n-----------")
        train_loss, train_acc = train(train_data_loader, model, loss_fn, optimizer)
        val_loss, val_acc = val(val_data_loader, model, loss_fn)

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        # 保存最好的模型权重
        if val_acc > min_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model_')
            min_acc = val_acc
            print(f"save best model, 第{t + 1}轮")
            torch.save(model.state_dict(), 'save_model_/best_model.pth')
        # 保存最后一轮的权重文件
        if t == epoch - 1:
            torch.save(model.state_dict(), 'save_model_/last_model.pth')
        lr_scheduler.step()

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('Done!')
