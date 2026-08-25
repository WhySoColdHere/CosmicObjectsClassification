import pandas as pd
import os
from PIL import Image, UnidentifiedImageError

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

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


df_without_aug = df[~df['filepath'].str.contains('_aug')]
images_info_series = df_without_aug['filepath'].apply(get_img_info)
images_info_df = pd.DataFrame([*images_info_series], columns=['filepath', 'image_width', 'image_height', 'image_mode', 'image_format', 'image_status'])
df_without_aug_images_info = pd.merge(left=df_without_aug, right=images_info_df, on='filepath')

print(df_without_aug_images_info.groupby('label')['image_status'].value_counts().reset_index().pivot(index='label', columns='image_status', values='count'))


