from datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from pytorch_metric_learning import losses, testers, distances, miners
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


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")


def test(train_set, test_set, model, accuracy_calculator, epoch):
    train_embeddings, train_labels = tester.get_all_embeddings(train_set, model)
    train_labels = train_labels.squeeze(1)
    test_embeddings, test_labels = tester.get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)

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


def make_datasets(model_name):
    train_dataset = TextDataset(
        data_path="/data/kaito_sugimoto/livedoor-corpus/text/livedoor_train.tsv",
        tokenizer_model_name=model_name
    )

    test_dataset = TextDataset(
        data_path="/data/kaito_sugimoto/livedoor-corpus/text/livedoor_test.tsv",
        tokenizer_model_name=model_name
    )
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    batch_size = 32
    lr = 0.01
    margin = 0.2
    num_epochs = 10
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = make_datasets(model_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Net(model_name=model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    distance = distances.CosineSimilarity()
    loss_func = losses.TripletMarginLoss(margin = margin, distance = distance)
    mining_func = miners.TripletMarginMiner(margin=margin, distance = distance, type_of_triplets="semihard")
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)


    test(train_dataset, test_dataset, model, accuracy_calculator, epoch=0)

    for epoch in range(1, num_epochs+1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        test(train_dataset, test_dataset, model, accuracy_calculator, epoch)