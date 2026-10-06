import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataCleaner:
    def __init__(self, path_to_dataset: str, split_seed=1) -> None:
        self.path_to_dataset = path_to_dataset
        self.split_seed = split_seed

    def __get_data(self):
        path_to_csv_df = os.path.join(self.path_to_dataset, 'ukidss_catalog.csv')
        path_to_img_folder = os.path.join(self.path_to_dataset, 'ukidss_final')

        df = pd.read_csv(path_to_csv_df)
        df['filepath'] = df['subfolder'] + r'/' + df['filename']
        df['filepath'] = df['filepath'].apply(lambda path_img, path_f: os.path.join(path_f, path_img),
                                              args=(path_to_img_folder,))
        df = df.drop(['subfolder', 'filename'], axis=1)

        return df

    def get_split_data(self):
        clean_df = self.__get_data()

        train_df, temp_df = train_test_split(clean_df, train_size=0.7, random_state=self.split_seed)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=self.split_seed)

        return {'train': train_df, 'validation': validation_df, 'test': test_df}
