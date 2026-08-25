from PIL import Image

class CosmicDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return Image.open(self.dataset['filepath'].iloc[idx]), self.dataset['label'].iloc[idx]
