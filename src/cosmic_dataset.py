from PIL import Image
from torch.utils.data import Dataset


class CosmicDataset(Dataset):
    def __init__(self, dataset, labels, transform):
        self.dataset = dataset
        self.labels = {v: k for (k, v) in enumerate(labels)}
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.dataset['filepath'].iloc[idx])
        image = self.transform(image)
        return image, self.labels[self.dataset['label'].iloc[idx]]
