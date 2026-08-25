import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)


class DataCleaner:
    def __init__(self, path_to_dataset: str, split_seed=1) -> None:
        self.path_to_dataset = path_to_dataset
        self.labels = os.listdir(self.path_to_dataset)
        self.split_seed = split_seed

        self.get_split_data()

    def __get_pseudo_df(self):
        pseudo_df = {'filepath': [], 'label': []}
        for label in self.labels:
            for file in os.listdir(os.path.join(self.path_to_dataset, label)):
                pseudo_df['filepath'].append(os.path.join(self.path_to_dataset, label, file))
                pseudo_df['label'].append(label)
        return pseudo_df

    def __get_clean_data(self):
        def get_img_info(image_path):
            image = image_width = image_height = image_mode = image_format = image_status = None

            try:
                image = Image.open(image_path)
                image_width, image_height = image.size
                image_mode = image.mode
                image_format = image.format
                image_status = 'ok'
            except UnidentifiedImageError:
                image_status = 'broken'
            finally:
                return image_path, image_width, image_height, image_mode, image_format, image_status

        dirty_df = pd.DataFrame(self.__get_pseudo_df())
        dirty_df = dirty_df.drop_duplicates()

        df_without_aug = dirty_df[~dirty_df['filepath'].str.contains('_aug')]
        images_info_series = df_without_aug['filepath'].apply(get_img_info)
        images_info_df = pd.DataFrame([*images_info_series],
                                      columns=['filepath', 'image_width', 'image_height', 'image_mode', 'image_format',
                                               'image_status'])
        df_without_aug_images_info = pd.merge(left=df_without_aug, right=images_info_df, on='filepath')
        clean_df = df_without_aug_images_info[df_without_aug_images_info['image_status'] == 'ok']

        return clean_df

    def get_split_data(self):
        clean_df = self.__get_clean_data()
        clean_df = clean_df.drop(['image_width', 'image_height', 'image_mode', 'image_format', 'image_status'], axis=1)


        train_df, temp_df = train_test_split(
            clean_df,
            train_size=0.7,
            random_state=self.split_seed,
            stratify=clean_df['label']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.split_seed,
            stratify=temp_df['label']
        )


        return {'train': train_df, 'validation': val_df, 'test': test_df, 'labels': clean_df['label'].unique()}



