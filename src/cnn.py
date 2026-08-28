import torch
from pandas import Series
from cosmic_dataset import CosmicDataset
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn


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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.to(self.device)

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

    def see_the_world_my_child(self):
        x_batch_img, y_batch_lb = next(iter(self.train_loader))

        self.optimizer.zero_grad()
        x_batch_img = x_batch_img.to(self.device)
        y_batch_lb = y_batch_lb.to(self.device)

        output = self(x_batch_img)
        loss = self.criterion(output, y_batch_lb)

        loss.backward()

        self.optimizer.step()

        print(loss)
        print(loss.shape)
