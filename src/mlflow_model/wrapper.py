import mlflow.pyfunc
from item_clustering.item_clustering import ItemClustering


class ItemClusteringWrapper:

    def __init__(self, model):
        self.item_clustering = model

    def predict(self, model_input):

        items = model_input.values.tolist()
        return self.item_clustering.predict(items)


# Define the model class
class ItemClustering(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        artifacts_path = context.artifacts['artifacts_path'][-1]

        self.item_clustering = ItemClustering()
        self.item_clustering.load_model(artifacts_path)


    def predict(self, context, model_input):

        items = model_input.values.tolist()
        return self.item_clustering.predict(items)
