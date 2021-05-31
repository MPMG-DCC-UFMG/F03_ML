import json


class Config:

    def __init__(self, word_embeddings_path="../dados/embeddings/fasttext/skip_s100.txt",
                 algorithm='hdbscan', categories=None, tags=None, operation='mean',
                 n_process=4, artifacts_path="../dados/output/"):

        self.word_embeddings_path = word_embeddings_path
        self.algorithm = algorithm
        self.categories = categories
        self.tags = tags
        self.operation = operation
        self.n_process = n_process
        self.artifacts_path = artifacts_path


    def get_config_dict(self):

        config_dict = {
            'word_embeddings_path' : self.word_embeddings_path,
            'algorithm' : self.algorithm,
            'categories' : self.categories,
            'tags' : self.tags,
            'operation' : self.operation,
            'n_process' : self.n_process,
            'artifacts_path' : self.artifacts_path
        }

        return config_dict


    def load_config(self, path):

        self.artifacts_path = path

        with open(self.artifacts_path + 'config.json', 'r') as config_file:
            config_dict = json.load(config_file)

        self.word_embeddings_path = config_dict['word_embeddings_path']
        self.algorithm = config_dict['algorithm']
        self.categories = config_dict['categories']
        self.tags = config_dict['tags']
        self.operation = config_dict['operation']
        self.n_process = config_dict['n_process']
        self.artifacts_path = config_dict['artifacts_path']


    def save_config(self, path):

        config_dict = self.get_config_dict()

        with open(self.artifacts_path + 'config.json', 'w') as config_file:
            json.dump(config_dict, config_file)
