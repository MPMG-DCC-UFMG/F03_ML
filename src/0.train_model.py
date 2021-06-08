from item_clustering.item_clustering import ItemClustering
from mlflow_model import conda_env
import mlflow_model.item_clustering
import pandas as pd
from utils.read_files import (
    get_items
)
import mlflow
from mlflow_model import wrapper
import time


def main():

    mlflow.set_experiment(experiment_name='banco_precos')





if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
