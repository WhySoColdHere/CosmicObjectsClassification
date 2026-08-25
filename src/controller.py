from conf import DATASET_PATH
from data_cleaning import DataCleaner
from cnn import CNN


class Controller:
    def __init__(self):
        cnn = CNN(DataCleaner(DATASET_PATH).get_split_data())
        cnn.see_the_world_my_child()

controller = Controller()

