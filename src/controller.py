from conf import DATASET_PATH
from data_cleaning import DataCleaner
from cnn import CosmicCNN
import time

class Controller:
    def __init__(self):
        # start = time.perf_counter()
        # cnn = CosmicCNN(DataCleaner(DATASET_PATH).get_split_data())
        # cnn.see_the_world_my_child()
        # end = time.perf_counter()
        #
        # print(f"Elapsed time: {end - start}")


        cnn = CosmicCNN(DataCleaner(DATASET_PATH).get_split_data())
        cnn.diagnose()

controller = Controller()






