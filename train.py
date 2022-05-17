import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

import model
from model import resnet
# from model import retinanet
# from model.retinanet import ResNet

import data
from data.Kitti import Kitti
import data.func
from data.func import Cropper, Resizer, Normalizer, collater

import dataloader

import loss
from loss import focalloss

import matplotlib.pyplot as plt

from skimage import io, transform

print("import completed.")


def train(dataset_train, dataset_val, params, num_epochs, loss, model):
    net = models.resnet18(pretrained=True)
    net = net.cuda() if device else net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    def accuracy(out, labels):
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 128)
    net.fc = net.fc.cuda() if use_cuda else net.fc

    n_epochs = 5
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(test_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

def main(args=None):
    dataset = Kitti(root="..")
    sample = dataset[0]

    image = sample["image"]
    print("image shape: ", image.shape)
    base_3Dto2D = sample["target"][0]["base"]
    print(base_3Dto2D.shape)
    plt.scatter(base_3Dto2D[0], base_3Dto2D[1])
    # plt.show()
    print("DEBUG")
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [len(dataset) - 2000, 2000])
    print("length of train: ", len(dataset_train))

    p = transforms.Compose([Resizer(224, 224), Normalizer()])
    p = transforms.Compose([Cropper(350, 1200), Normalizer()])
    pData = p(sample)
    image = pData["image"]
    labels = pData["labels"]
    print("image shape: ", image.shape)
    print(image.mean())
    print(labels.shape)
    print(labels)

    dataloader_train = None
    dataloader_val = None

    use_gpu = False

    train_dataset = Kitti(root="..", transforms=p)
    main_model = resnet.resnet18()
    # main_model.fc = nn.Linear(512, 2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=3,
        shuffle=True,
        collate_fn=collater
    )
    batch = next(iter(train_loader))
    print(batch["image"].shape)
    print(batch["labels"].shape)
    output = main_model(batch["image"])
    print(output.shape)
    # train(dataset_train, dataset_val, None, 20, focalloss, main_model)

    for iter_num, data in enumerate(train_loader):
        print(".")

if __name__ == '__main__':
    main()