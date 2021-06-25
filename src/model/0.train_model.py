
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import time
import argparse
from item_clustering.item_clustering import ItemClustering
from mlflow_model import conda_env
import mlflow_model.item_clustering
import pandas as pd
from utils.read_files import (
    get_items
)
import mlflow
from mlflow_model import wrapper
from util.get_items_hive import get_items_hive


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('-x', '--experiment_name', type=str,
                  default='banco_precos', help='name of the experiment.')
    p.add_argument('--hive', type=str, default='',
                    help='input hive table.')
    p.add_argument('-i', '--input', type=str,
                  default='../data/dataset_item_druid.csv', help='items table.')
    p.add_argument('-o', '--outpath', type=str, default='../data/output/test/',
                   help='path to the write the outputs')
    p.add_argument('-e', '--embeddings_path',type=str,
        default='../data//embeddings/models/fasttext/sg/output/items_embeddings.vec',
        help='path to the file containing the embeddings to be used in the representation')
    p.add_argument('--n_process', type=int, default=6,
                   help='number of process to use on clustering')

    parsed = p.parse_args()

    return parsed


def main():

    args = parse_args()

    config = {
        'artifacts_path' : args.outpath,
        'word_embeddings_path' : args.embeddings_path,
        'n_process' : args.n_process
    }

    model = ItemClustering(config=config)

    file = args.input
    if bool(args.hive):
        items = get_items(file)
    else:
        items = get_items_hive(args.hive)

    mlflow.set_experiment(experiment_name=args.experiment_name)
    mlflow_pyfunc_model_path = "item_clustering_mlflow_pyfunc"

    with mlflow.start_run(run_name='ItemClustering'):
        mlflow.log_params(model.config.get_config_dict())

        print('[1] Training model...')
        model.fit(items)

        print('[2] Saving model...')
        model.save_model()

        print('[3] Evaluating model...')
        metrics = model.evaluate()

        # log metrics
        mlflow.log_metrics(metrics)

        artifacts = {
            "artifacts_path": model.config.artifacts_path
        }
        mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=wrapper.ItemClustering(),
                                artifacts=artifacts, conda_env=conda_env.conda_env)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
