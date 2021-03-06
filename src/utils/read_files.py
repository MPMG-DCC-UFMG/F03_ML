# imports

import pandas as pd
import json
import pickle


def get_items(file_items):

    data = pd.read_csv(file_items, sep=';', low_memory=False)
    items = data[['nome_item', 'id_hom_licitacao', 'id_item_licitacao',
                  'id_licitacao', 'vlr_unitario_homologado', 'unidade_medida',
                  'ano']].values.tolist()

    return items


def read_json_file(file):

    with open(file, "r") as JFile:
        data = json.load(JFile)
    JFile.close()

    return data


def read_pickle_file(file):

    with open(file, "rb") as PFile:
        data = pickle.load(PFile)
    PFile.close()

    return data


def save_json_file(file, data):

    with open(file, "w") as JFile:
        json.dump(data, JFile)
    JFile.close()


def save_pickle_file(file, data):

    with open(file, "wb") as PFile:
        pickle.dump(data, PFile)
    PFile.close()
