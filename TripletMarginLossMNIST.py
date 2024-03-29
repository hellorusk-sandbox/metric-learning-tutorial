from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"shape of data: {data.shape}")
            print(f"shape of embeddings: {embeddings.shape}")
            print(f"shape of labels: {labels.shape}")
            print(f"Epoch {epoch} Iteration {batch_idx}: Loss = {loss}")


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator, epoch):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")

    # 第2引数は retrieve される最近傍の方、第1引数はクエリ
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                test_embeddings,
                                                test_labels,
                                                test_labels,
                                                embeddings_come_from_same_source=True)
    print(f"Test set accuracy (Precision@1) = {accuracies['precision_at_1']}")

    train_embeddings_reduced = TSNE(n_components=2, random_state=0).fit_transform(train_embeddings.cpu())
    plt.figure(figsize=(8, 8))
    plt.scatter(train_embeddings_reduced[:, 0], train_embeddings_reduced[:, 1], c=train_labels.cpu(), cmap=cm.tab10)
    plt.savefig(f"{__file__}_epoch{epoch}_train.png")

    test_embeddings_reduced = TSNE(n_components=2, random_state=0).fit_transform(test_embeddings.cpu())
    plt.figure(figsize=(8, 8))
    plt.scatter(test_embeddings_reduced[:, 0], test_embeddings_reduced[:, 1], c=test_labels.cpu(), cmap=cm.tab10)
    plt.savefig(f"{__file__}_epoch{epoch}_test.png")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 256

dataset1 = datasets.MNIST('.', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('.', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=256)

# check data
for batch_idx, (data, labels) in enumerate(train_loader):
    if batch_idx == 0:
        print(data.shape)
        print(data)
        print(labels.shape)
        print(labels)

for batch_idx, (data, labels) in enumerate(test_loader):
    if batch_idx == 0:
        print(data.shape)
        print(data)
        print(labels.shape)
        print(labels)


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1

loss_func = losses.TripletMarginLoss(margin = 0.05)
accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

test(dataset1, dataset2, model, accuracy_calculator, epoch=0)

for epoch in range(1, num_epochs+1):
    train(model, loss_func, device, train_loader, optimizer, epoch)
    test(dataset1, dataset2, model, accuracy_calculator, epoch)