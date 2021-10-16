from datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from pytorch_metric_learning import losses, testers, distances, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer_model_name, max_length=512):
        super(TextDataset, self).__init__()
        self.data = []
        self.label = []

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

        with open(data_path, "r") as f:
            for line in f.readlines():
                split_line = line.split("\t")
                text = split_line[0]
                input_ids = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')[0]
                label = int(split_line[1].replace("\n", ""))
                self.data.append(input_ids)
                self.label.append(label)

        self.data = torch.stack(self.data)
        self.label = torch.Tensor(self.label)
        self.label = self.label.to(torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class BertEncoder(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertEncoder, self).__init__()

        self.bert_model = bert_model

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert = self.bert_model(
            token_ids, segment_ids, attention_mask
        )

        embeddings = output_bert["last_hidden_state"][:, 0, :] # [CLS] embeddings
        return embeddings


class BertEncoderModule(torch.nn.Module):
    def __init__(self, model_name):
        super(BertEncoderModule, self).__init__()

        bert = AutoModel.from_pretrained(model_name)
        self.encoder = BertEncoder(bert)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(self.device)

        self.NULL_IDX = 0

    def to_bert_input(self, token_idx, null_idx):
        segment_idx = token_idx * 0
        mask = token_idx != null_idx
        mask = mask.long()
        token_idx = token_idx * mask
        return token_idx, segment_idx, mask

    def forward(
        self,
        input_ids,
    ):
        token_idx, segment_idx, mask = self.to_bert_input(
            input_ids, self.NULL_IDX
        )
        embeddings = self.encoder(
            token_idx, segment_idx, mask
        )

        return embeddings


def train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch):
    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} Iteration {batch_idx}: Loss = {loss}, Number of mined triplets = {mining_func.num_triplets}")


def test(train_set, test_set, model, accuracy_calculator, epoch):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    train_labels = train_labels.squeeze(1)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
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


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def make_datasets(model_name, max_length=512):
    train_dataset = TextDataset(
        data_path="/data/kaito_sugimoto/livedoor-corpus/text/livedoor_train.tsv",
        tokenizer_model_name=model_name,
        max_length=max_length
    )

    test_dataset = TextDataset(
        data_path="/data/kaito_sugimoto/livedoor-corpus/text/livedoor_test.tsv",
        tokenizer_model_name=model_name,
        max_length=max_length
    )
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    batch_size = 128
    lr = 0.01
    margin = 1
    num_epochs = 10
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    max_length = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = make_datasets(model_name, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


    model = BertEncoderModule(model_name=model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    distance = distances.LpDistance(normalize_embeddings=False)
    loss_func = losses.ArcFaceLoss(num_classes=9, embedding_size=max_length).to(device)
    loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=lr)
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)


    test(train_dataset, test_dataset, model, accuracy_calculator, epoch=0)

    for epoch in range(1, num_epochs+1):
        train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch)
        test(train_dataset, test_dataset, model, accuracy_calculator, epoch)