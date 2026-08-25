from pandas import Series
from cosmic_dataset import CosmicDataset
import torchvision
from torch.utils.data import DataLoader

class CNN:
    def __init__(self, dataset: Series):
        self.batch_size = 32
        self.common_img_size = (128, 128)
        self.labels = dataset['labels']

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

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

    def see_the_world_my_child(self):
        images, labels = next(iter(self.train_loader))
        print(images[0].shape)
        print(self.train_dataset[0][0].shape)