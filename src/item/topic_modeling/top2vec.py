import numpy as np
import pandas as pd
import multiprocessing
from item.clustering.utils import *
from item.clustering.clustering import run_dim_reduction
from nlp.word_embeddings import calc_distance

# Import UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction)
import umap


def get_cluster_centroid(group_desc, items_vec):

    embeddings_size = len(list(items_vec.values())[0])
    centroid = np.zeros(embeddings_size)
    num_items = len(group_desc)

    for item in group_desc:
        centroid += np.array(items_vec[str(item)])

    centroid /= num_items

    return centroid


def get_token_embeddings(items_data, group_desc, word_embeddings,
                         dimred_model=None):

    tokens = set()

    for desc_id in group_desc:
        item_dict = items_data.loc[desc_id].to_dict()
        if isinstance(item_dict['original_prep'], str):
            description = eval(item_dict['original_prep'])
        else:
            description = item_dict['original_prep']
        for token in description:
            tokens.add(token)

    tokens = list(tokens)
    tok_embeddings = []
    tokens_with_embedding = []

    for token in tokens:
        if token in word_embeddings:
            tokens_with_embedding.append(token)
            tok_embedding = np.array(word_embeddings[token])
            if dimred_model is not None:
                tok_embedding += [0.0] * 100
            tok_embeddings.append(np.array(tok_embedding))

    if dimred_model is not None:
        embeddings_matrix = dimred_model.transform(tok_embeddings)
    elif len(tok_embeddings) > 15:
        reducer = run_dim_reduction(tok_embeddings, algorithm='UMAP', init='random')
        embeddings_matrix = reducer.transform(tok_embeddings)
    else:
        return tokens

    return dict(zip(tokens_with_embedding, embeddings_matrix))


def find_topic_words_and_scores(centroid, tok_embedding, num_words,
                                distance='cosine'):

    tokens_score = []

    for token, embedding in tok_embedding.items():
        score = calc_distance(centroid, embedding, distance=distance)
        tokens_score.append((token, score))

    tokens_score.sort(key=lambda x: x[1])
    tokens_score = tokens_score[:num_words]

    return tokens_score


def top2vec(items_data, groups, word_embeddings, items_vec, it_process,
            results_dict, reducer_model=None, distance='cosine', num_words=10):

    print(it_process)

    # It creates a list of the the keys of these groups:
    groups_names = groups[0]

    # It gets the values of each group (i.e., the ids of the descriptions into that group):
    group_descriptions = groups[1]
    cluster_words = {}

    for ft_it in range(len(groups_names)):
        group_name = groups_names[ft_it]
        first_token = group_name.split('_')[0]
        dimred_model = None
        if reducer_model is not None:
            dimred_model = reducer_model[first_token]
        tok_embedding = get_token_embeddings(items_data, group_descriptions[ft_it],
                                              word_embeddings, dimred_model)
        if isinstance(tok_embedding, list):
            words = [(tok, 1.0) for tok in tok_embedding]
        else:
            centroid = get_cluster_centroid(group_descriptions[ft_it], items_vec)
            words = find_topic_words_and_scores(centroid, tok_embedding, num_words,
                                                distance=distance)
        cluster_words[group_name] = words

    results_dict[it_process] = cluster_words


def merge_multiprocessing_results(results_process, n_process):

    cluster_words = {}

    for i in range(n_process):
        cluster_words.update(results_process[i])

    return cluster_words


def get_valid_groups(groups):

    valid_groups = {}

    for group_name, desc_ids in groups.items():
        if '_' in group_name and "-1" not in group_name:
            valid_groups[group_name] = desc_ids

    return valid_groups


def get_cluster_words(itemlist, groups, word_embeddings, items_vec,
                      reducer_model=None, distance='cosine', num_words=10,
                      n_process=4):

    manager = multiprocessing.Manager()
    results_process = manager.dict()
    jobs = []
    groups = get_valid_groups(groups)

    # It defines the ranges (of the groups) the process will work on:
    group_len = len(groups)
    process_ranges = get_ranges(group_len, n_process)
    print('Read ranges')
    print(process_ranges)

    process_items = get_items_for_processes(itemlist.items_df, n_process,
                                            process_ranges, groups)
    del itemlist

    for i in range(n_process):
        items_data = process_items[i][0]
        groups_names = process_items[i][1]
        groups_items = process_items[i][2]

        p = multiprocessing.Process(target=top2vec,
            args = (items_data, (groups_names, groups_items), word_embeddings,
                    items_vec, i, results_process, reducer_model, distance,
                    num_words))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        proc.close()

    cluster_words = merge_multiprocessing_results(results_process, n_process)

    return cluster_words
