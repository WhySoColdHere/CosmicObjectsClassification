from conf import DATASET_PATH
from data_cleaning import DataCleaner
from cnn import CosmicCNN
from datetime import datetime

class Controller:
    def __init__(self):
        start = datetime.now()
        cnn = CosmicCNN(DataCleaner(DATASET_PATH).get_split_data())
        cnn.see_the_world_my_child()
        end = datetime.now()

        print(f"Elapsed time: {end - start}")
controller = Controller()






