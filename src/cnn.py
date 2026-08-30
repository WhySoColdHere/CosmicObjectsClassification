import time
import torch
from pandas import Series
from cosmic_dataset import CosmicDataset
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix



class CosmicCNN(nn.Module):
    def __init__(self, dataset: Series):
        super().__init__()
        self.batch_size = 32
        self.common_img_size = (128, 128)
        self.labels = dataset['labels']
        self.epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(65536, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer_out = nn.Linear(128, 8)
        self.criterion = nn.CrossEntropyLoss()

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self._create_transforms()
        self._create_datasets(dataset)
        self._create_dataloaders()

    def _create_transforms(self):
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.common_img_size),
            torchvision.transforms.ToTensor()
        ])
        self.val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.common_img_size),
            torchvision.transforms.ToTensor()
        ])
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.common_img_size),
            torchvision.transforms.ToTensor()
        ])

    def _create_datasets(self, dataset):
        self.train_dataset = CosmicDataset(dataset['train'], self.labels, self.train_transform)
        self.validation_dataset = CosmicDataset(dataset['validation'], self.labels, self.val_transform)
        self.test_dataset = CosmicDataset(dataset['test'], self.labels, self.test_transform)

    def _create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer_out(x)

        return x

    def _train(self):
        self.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for x_batch_img, y_batch_lb in self.train_loader:
                self.optimizer.zero_grad()
                x_batch_img = x_batch_img.to(self.device)
                y_batch_lb = y_batch_lb.to(self.device)

                output = self(x_batch_img)
                loss = self.criterion(output, y_batch_lb)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(self.train_loader)
            val_accuracy, val_loss = self._validation()

            print(
                f"Epoch {epoch + 1} / {self.epochs} "
                f"| train loss: {train_loss:4f} "
                f" validation loss: {val_loss:4f} "
                f" validation accuracy: {val_accuracy:4f}"
            )

    def _validation(self):
        self.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch_img, y_batch_lb in self.val_loader:
                x_batch_img = x_batch_img.to(self.device)
                y_batch_lb = y_batch_lb.to(self.device)

                output = self(x_batch_img)
                loss = self.criterion(output, y_batch_lb)

                total_loss += loss.item()

                predictions = output.argmax(dim=1)
                correct += (predictions == y_batch_lb).sum().item()
                total += y_batch_lb.size(0)

        average_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return accuracy, average_loss

    def _test(self):
        self.eval()

        all_predictions = []
        all_targets = []

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch_img, y_batch_lb in self.test_loader:
                x_batch_img = x_batch_img.to(self.device)
                y_batch_lb = y_batch_lb.to(self.device)

                output = self(x_batch_img)
                loss = self.criterion(output, y_batch_lb)

                total_loss += loss.item()

                predictions = output.argmax(dim=1)
                correct += (predictions == y_batch_lb).sum().item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch_lb.cpu().numpy())
                total += y_batch_lb.size(0)

        average_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        matrix = confusion_matrix(all_targets, all_predictions)

        return accuracy, average_loss, matrix

    def see_the_world_my_child(self):
        print("Starting training")
        self._train()
        test_accuracy, test_loss, conf_matrix = self._test()

        print(
            f"Test accuracy: {test_accuracy:4f}\n"
            f"Test loss: {test_loss:4f}\n"
            f"Confusion matrix:\n{conf_matrix}"
        )

    def diagnose(self):
        start = time.perf_counter()

        for batch_counter, (x, y) in enumerate(self.train_loader):
            if batch_counter >= 10:
                break

        end = time.perf_counter()

        print(f"10 batches loading time: {end - start}")