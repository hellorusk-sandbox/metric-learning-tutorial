from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.legacy import data
from transformers import AutoTokenizer, AutoModel

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

tester = testers.BaseTester()


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer_model_name):
        super(TextDataset, self).__init__()
        self.data = []
        self.label = []

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

        with open(data_path, "r") as f:
            for line in f.readlines():
                split_line = line.split("\t")
                text = split_line[0]
                input_ids = tokenizer.encode(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')[0]
                label = int(split_line[1].replace("\n", ""))
                self.data.append(input_ids)
                self.label.append(label)

        self.data = torch.stack(self.data)
        self.label = torch.Tensor(self.label)
        self.label = self.label.to(torch.int8)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class Net(torch.nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids):
        output = self.model(input_ids)
        embeddings = output["last_hidden_state"][:, 0, :]
        return embeddings


def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        data = batch.text
        labels = batch.labels
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}: Loss = {loss}")


def test(test_set, model, accuracy_calculator):
    test_embeddings, test_labels = tester.get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)

    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                test_embeddings,
                                                test_labels,
                                                test_labels,
                                                embeddings_come_from_same_source=True)
    print(f"Test set accuracy (Precision@1) = {accuracies['precision_at_1']}")

    test_embeddings_reduced = TSNE(n_components=2, random_state=0).fit_transform(test_embeddings)
    plt.scatter(test_embeddings_reduced[:, 0], test_embeddings_reduced[:, 1], c=test_labels)


def make_datasets(model_name):
    dataset = TextDataset(
        data_path="/data/kaito_sugimoto/livedoor-corpus/text/livedoor_head100.tsv",
        tokenizer_model_name=model_name
    )

    hoge_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for i, (x, y) in enumerate(hoge_loader):
        if i == 0:
            print(x)
            print(y)

    # WIP!
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    batch_size = 16
    lr = 0.01
    margin = 0.05
    num_epochs = 10
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = make_datasets(model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for i, (x, y) in enumerate(train_loader):
        if i == 0:
            print(x)
            print(y)

    model = Net(model_name=model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_func = losses.TripletMarginLoss(margin = margin)
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)


    test(test_dataset, model, accuracy_calculator)

    for epoch in range(1, num_epochs+1):
        train(model, loss_func, device, train_loader, optimizer, epoch)
        test(test_dataset, model, accuracy_calculator)