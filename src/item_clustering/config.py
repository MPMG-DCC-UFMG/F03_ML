import json


class Config:

    def __init__(self, word_embeddings_path="../data/embeddings/fasttext/skip_s100.txt",
                 algorithm='hdbscan', categories=['unidades_medida', 'numeros'],
                 tags=['N', 'MED'], operation='concatenate', n_process=4,
                 spellcheck="../data/dicionario/replacement_licitacao.json",
                 artifacts_path="../data/output/", regrouping=True):

        self.word_embeddings_path = word_embeddings_path
        self.spellcheck = spellcheck
        self.algorithm = algorithm
        self.categories = categories
        self.tags = tags
        self.operation = operation
        self.n_process = n_process
        self.artifacts_path = artifacts_path
        self.regrouping = regrouping


    def get_config_dict(self):

        config_dict = {
            'word_embeddings_path' : self.word_embeddings_path,
            'spellcheck' : self.spellcheck,
            'algorithm' : self.algorithm,
            'categories' : self.categories,
            'tags' : self.tags,
            'operation' : self.operation,
            'n_process' : self.n_process,
            'artifacts_path' : self.artifacts_path,
            'regrouping' : self.regrouping
        }

        return config_dict


    def load_config(self, path):

        self.artifacts_path = path

        with open(self.artifacts_path + 'config.json', 'r') as config_file:
            config_dict = json.load(config_file)

        self.word_embeddings_path = config_dict['word_embeddings_path']
        self.spellcheck = config_dict['spellcheck']
        self.algorithm = config_dict['algorithm']
        self.categories = config_dict['categories']
        self.tags = config_dict['tags']
        self.operation = config_dict['operation']
        self.n_process = config_dict['n_process']
        self.artifacts_path = config_dict['artifacts_path']
        self.regrouping = config_dict['regrouping']


    def save_config(self, path):

        config_dict = self.get_config_dict()

        with open(self.artifacts_path + 'config.json', 'w') as config_file:
            json.dump(config_dict, config_file)
