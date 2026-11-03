import torch
from PIL import Image
from torch.utils.data import Dataset
from funcs_storage import FuncsStorage


class CosmicDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.questions_answers = FuncsStorage.get_dataset_structure(dataset.columns)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = Image.open(self.dataset['filepath'].iloc[idx])
        img = self.transform(img)
        target, mask = self.__process_dataset(self.dataset.iloc[idx])

        return img, target, mask

    def __process_dataset(self, row):
        target = dict()
        mask = dict()
        for i in self.questions_answers:
            row_values = row[self.questions_answers[i]].values
            target[i] = torch.tensor(row_values.astype('float32'))
            mask[i] = row_values.sum() > 0
        return target, mask
