"""
The ``mlflow_model.ItemClustering`` module provides an API for logging and loading ItemClustering models.
      This module exports ItemClustering models with the following flavors:
ItemClustering (native) format
    This is the main flavor that can be loaded back into ItemClustering.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import yaml
import pandas as pd

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

from item_clustering.item_clustering import ItemClustering
from .wrapper import ItemClusteringWrapper
import mlflow_model.item_clustering

FLAVOR_NAME = "ItemClustering"

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "hdbscan",
        "umap-learn"
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)


def save_model(item_clustering_model, path='./', conda_env=None, mlflow_model=Model()):
    """
    Save a ItemClustering model to a path on the local file system.
    :param item_clustering_model: ItemClustering model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file.
    :param mlflow_model: MLflow model config this flavor is being added to.
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)

    item_clustering_model.save_model()
    model_data_subpath = item_clustering_model.config.artifacts_path

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow_model.item_clustering",
                        data=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(item_clustering_model, artifact_path, conda_env=None, **kwargs):
    """
    Log a ItemClustering model as an MLflow artifact for the current run.
    :param item_clustering_model: ItemClustering model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow_model.item_clustering,
              item_clustering_model=item_clustering_model, conda_env=conda_env,
              **kwargs)


def _load_model(model_file):

    item_clustering = ItemClustering()
    item_clustering.load_model(model_file)

    return item_clustering


def _load_pyfunc(model_file):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    model = _load_model(model_file)
    return ItemClusteringWrapper(model)


def load_model(path, run_id=None):
    """
    Load a ItemClustering model from a local file (if ``run_id`` is None) or a run.
    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.item_clustering.log_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    path = os.path.abspath(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    # Flavor configurations for models saved in MLflow version <= 0.8.0 may not contain a
    # `data` key; in this case, we assume the model artifact path to be `model.h5`
    item_clustering_model_artifacts_path = os.path.join(path)
    return _load_model(model_file=item_clustering_model_artifacts_path)
