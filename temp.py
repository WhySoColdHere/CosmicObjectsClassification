from PIL import Image, UnidentifiedImageError
import pandas as pd
import os

pd.set_option('display.max_columns', None)

df = pd.read_csv('data/ukidss_catalog.csv')

# NaN отсутствует: df.isna().sum().sum()
# Дубликаты отсутствуют: df.duplicated().sum()
# Типы данных соответствуют: df.info() и df.dtypes

img_info = {
    "path": [],
    "width": [],
    "height": [],
    "format": [],
    "mode": [],
    "status": [],
}


def check_image(path, res_df):
    try:
        img = Image.open(path)
        res_df['width'].append(img.size[0])
        res_df['height'].append(img.size[1])
        res_df['format'].append(img.format)
        res_df['mode'].append(img.mode)
        res_df['status'].append('ok')
    except UnidentifiedImageError:
        res_df['status'].append('broken')
    finally:
        res_df['path'].append(path)


path_to_main_folder = os.path.join('./', 'data', 'ukidss_final')
img_folder_path = pd.Series(df['subfolder'] + '/' + df['filename'])
create_path = lambda path_img, path_f: os.path.join(path_f, path_img)
img_completed_path = img_folder_path.apply(create_path, args=(path_to_main_folder,))
img_completed_path.head(500).apply(check_image, args=(img_info,)).fillna('broken')
img_info_df = pd.DataFrame.from_dict(img_info)

learning_columns = df.columns[2:]
separator = '-ukidss_'
column_start = lambda x: x[:x.find(separator)]
column_end = lambda x: x[x.find(separator) + len(separator):]
questions_answers = dict((q, {'votes': [], 'fractions': []}) for q in map(column_start, learning_columns))

for col in learning_columns:
    column = column_start(col)
    if col.endswith('fraction'):
        questions_answers[column]['fractions'].append(column_end(col))
    else:
        questions_answers[column]['votes'].append(column_end(col))

reconstruct_columns = lambda col_list, prefix: [prefix + separator + i for i in col_list]
get_columns = lambda key: reconstruct_columns(questions_answers[key]['votes'], key) + reconstruct_columns(questions_answers[key]['fractions'], key)

print(df[get_columns('smooth-or-featured')].head(10))


