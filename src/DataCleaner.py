import pandas as pd
import os
from PIL import Image

class DataCleaner:
    def __init__(self, path_to_dataset: str) -> None:
        self.path_to_dataset = path_to_dataset
        self.labels = os.listdir(self.path_to_dataset)

        self._df = pd.DataFrame(self.__get_pseudo_df())

    def __get_pseudo_df(self):
        pseudo_df = {'filepath': [], 'label': []}
        for label in self.labels:
            for file in os.listdir(os.path.join(self.path_to_dataset, label)):
                pseudo_df['filepath'].append(os.path.join(self.path_to_dataset, label, file))
                pseudo_df['label'].append(label)
        return pseudo_df

    def get_clean_data(self):
        return self._df


path = os.path.join('..', 'data')
dc = DataCleaner(path)
df = dc.get_clean_data()


img = Image.open(df['filepath'].iloc[0])
print(img.size)
print(img.mode)
print(img.format)

