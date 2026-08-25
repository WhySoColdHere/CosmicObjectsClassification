from conf import DATASET_PATH
from data_cleaning import DataCleaner

from cosmic_dataset import CosmicDataset


class CNN:
    def __init__(self, dataset):
        train_dataset = CosmicDataset(dataset['train'])
        validation_dataset = CosmicDataset(dataset['validation'])
        test_dataset = CosmicDataset(dataset['test'])

        print(train_dataset[0])


cnn = CNN(DataCleaner(DATASET_PATH).get_split_data())
