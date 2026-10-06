import torch
from PIL import Image
from torch.utils.data import Dataset

class CosmicDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.learn_columns = [col for col in self.dataset.columns if col != 'filepath' and col.endswith('fraction')]
        self.transform = transform
        self.separator = '-ukidss_'
        self.column_prefix = lambda x: x[:x.find(self.separator)]
        self.questions_answers = dict((q, []) for q in map(self.column_prefix, self.learn_columns))
        for col in self.learn_columns:
            if col.endswith('fraction'):
                self.questions_answers[self.column_prefix(col)].append(col)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = Image.open(self.dataset['filepath'].iloc[idx])
        if self.transform:
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

