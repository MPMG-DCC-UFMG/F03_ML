# imports

import numpy as np
import pandas as pd
import random
import collections
from .utils import group_dsc_unidade_medida, add_first_token_column, add_outlier_column


def get_items_sample(itemlist, prices, items):
    '''
        Get a sample of the item prices. All values in the result sample are in
        the greater than the minimum value on the list 'prices' and less than
        the maximum value on the list 'prices'.

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        prices (list): list of item prices.
        items (list): list of item ids.
    '''

    new_items = []
    if len(prices) == 0:
        return new_items

    mini = np.min(prices)
    maxi = np.max(prices)
    for item in items:
        item_dict = itemlist.items_df.iloc[item].to_dict()
        if item_dict['preco'] >= mini and item_dict['preco'] <= maxi:
            new_items.append(item)

    return new_items


def get_items_prices(itemlist, items):
    '''
        Get item prices.

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        items (list): list of item ids.
    '''

    prices = []
    for item in items:
        prices.append(itemlist.items_df.iloc[item]['preco'])

    return prices


def get_clusters_prices(itemlist, results):
    '''
        Get item prices for each group in the set.

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        results (dict): dictionary of groups (group_id -> list of item ids).
    '''

    cluster_prices = {}

    for cluster, items in results.items():
        prices = get_items_prices(itemlist, items)
        cluster_prices[cluster] = prices

    return cluster_prices


def remove_outlier_prices(itemlist, cluster_items, cluster_prices, threshold,
                          baseline=True):
    '''
        Remove the outlier items for each group in the set. All items that have
        price less than 10th percentile or greater than 90th percentile of items
        are removed from the groups.

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        cluster_items (dict): dictionary of groups (group_id -> list of item ids).
        cluster_prices (dict): item prices for each group (group_id -> list of
                               item ids).
        threshold (float): all groups which have mean/std > threshold, have the
                           outliers removed.
        baseline (bool): if the first token grouping was used to generete the
                         groups.
    '''

    for cluster, prices in cluster_prices.items():
        if baseline and '_' not in cluster:
            continue
        elif not baseline and cluster == '-1':
            continue
        std = np.std(prices)
        mean = np.mean(prices)
        if std/mean > threshold:
            prices.sort()
            lower = np.percentile(prices, 10)
            upper = np.percentile(prices, 90)
            new_prices = [p for p in prices if p > lower and p < upper]
            cluster_prices[cluster] = new_prices
            cluster_items[cluster] = get_items_sample(itemlist, new_prices, \
                                                      cluster_items[cluster])

    return cluster_prices, cluster_items


def get_percentiles(prices):
    '''
        Compute the q-th percentile of the data along the specified axis.

        prices (list): item prices.
    '''

    percentiles = []
    for i in range(10, 100, 10):
        percentiles.append(np.percentile(prices, i))

    return percentiles


def sample_per_decile(itemlist, cluster_items, cluster_prices, threshold, baseline=True,
                     random_seed=999):
    '''
        Sample the items of each group per decile.

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        cluster_items (dict): dictionary of groups (group_id -> list of item ids).
        cluster_prices (dict): item prices for each group (group_id -> list of
                               item ids).
        threshold (float): all groups which have mean/std > threshold, have the
                           outliers removed.
        baseline (bool): if the first token grouping was used to generete the
                         groups.
    '''

    random.seed(random_seed)

    for cluster, prices in cluster_prices.items():
        if baseline and '_' not in cluster:
            continue
        elif not baseline and cluster == '-1':
            continue
        std = np.std(prices)
        mean = np.mean(prices)
        if std/mean > threshold:
            prices.sort()
            items = []
            percentiles = get_percentiles(prices)
            first_percentile = [p for p in prices if p <= percentiles[0]]
            items_sample = get_items_sample(itemlist, first_percentile, cluster_items[cluster])
            items += list(random.sample(items_sample, int(0.2*len(items_sample))))
            for i in range(1, len(percentiles)):
                lower = percentiles[i - 1]
                upper = percentiles[i]
                percentile = [p for p in prices if p > lower and p > upper]
                items_sample = get_items_sample(itemlist, percentile, cluster_items[cluster])
                items += list(random.sample(items_sample, int(0.2*len(items_sample))))
            last_percentile = [p for p in prices if p >= percentiles[-1]]
            items_sample = get_items_sample(itemlist, last_percentile, cluster_items[cluster])
            items += list(random.sample(items_sample, int(0.2*len(items_sample))))
            cluster_items[cluster] = items
            cluster_prices[cluster] = get_items_prices(itemlist, items)

    return cluster_prices, cluster_items


def get_items_dataframe(itemlist, cluster_items, baseline=True):
    '''
        Get the items dataframe. The dataframe should have the following columns:
        ['item_id', 'seq_dim_licitacao', 'outlier', 'cluster', 'dsc_unidade_medida',
        'description', 'price'].

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        cluster_items (dict): dictionary of groups (group_id -> list of item ids).
        baseline (bool): if the first token grouping was used to generete the
                         groups.
    '''

    group_dsc_unidade_medida(itemlist.items_df)

    lines = []
    for cluster, items in cluster_items.items():
        for item in items:
            item_dict = itemlist.items_df.iloc[item].to_dict()
            price = item_dict['preco']
            dsc_unidade_medida = item_dict['dsc_unidade_medida']
            year = item_dict['ano']
            try:
                dsc_unidade_medida = dsc_unidade_medida.lower()
            except:
                dsc_unidade_medida = ""
            description = ' '.join(eval(item_dict['original_prep']))
            original = item_dict['original']
            licitacao = item_dict['licitacao']
            areas = item_dict['funcao']
            if baseline and ('_' not in cluster or cluster[-2:] == '-1'):
                line = (item, licitacao, 1, cluster, dsc_unidade_medida, year, \
                        description, original, areas, price)
            elif not baseline and cluster == '-1':
                line = (item, licitacao, 1, cluster, dsc_unidade_medida, year, \
                        description, original, areas, price)
            else:
                line = (item, licitacao, 0, cluster, dsc_unidade_medida, year, \
                        description, original, areas, price)
            lines.append(line)

    columns = ('item_id', 'seq_dim_licitacao', 'outlier', 'cluster', \
               'dsc_unidade_medida', 'ano', 'description', 'original', 'areas', \
               'price')
    items_clusters_df = pd.DataFrame(lines, columns=columns)
    items_clusters_df = add_first_token_column(items_clusters_df)

    return items_clusters_df


def get_prices_statistics_df(items_clusters_df, dsc_unidade=True, year=False):
    '''
        Get item prices statistics such as mean, median, max, min, first
        quantile, etc.

        items_clusters_df (DataFrame): items dataframe. It should have the
                                        following columns:
        ['item_id', 'seq_dim_licitacao', 'outlier', 'cluster', 'dsc_unidade_medida',
        'description', 'price'].
        dsc_unidade (bool): if the field 'dsc_unidade_medida' should be used for
                            pricing.
        year (bool): if the field 'ano' should be used for pricing.
    '''

    if dsc_unidade and year:
        group_by = ['cluster', 'dsc_unidade_medida', 'ano']
    elif dsc_unidade:
        group_by = ['cluster', 'dsc_unidade_medida']
    elif year:
        group_by = ['cluster', 'ano']
    else:
        group_by = ['cluster']

    results_df = items_clusters_df[group_by + ['price']]

    results_grouped=results_df.groupby(group_by, as_index=False)['price'].mean()
    results_grouped=results_grouped.rename(columns = {'price':'mean'})
    results_grouped['count']=results_df.groupby(group_by, as_index=False)['price'].count().transform('price')
    results_grouped['max']=results_df.groupby(group_by, as_index=False)['price'].max().transform('price')
    results_grouped['min']=results_df.groupby(group_by, as_index=False)['price'].min().transform('price')
    results_grouped['median']=results_df.groupby(group_by, as_index=False)['price'].median().transform('price')
    results_grouped['std']=results_df.groupby(group_by)['price'].std().reset_index().transform('price')
    results_grouped['var']=results_df.groupby(group_by)['price'].var().reset_index().transform('price')
    results_grouped['quantile_1']=results_df.groupby(group_by)['price'].quantile(q=0.25).reset_index().transform('price')
    results_grouped['quantile_3']=results_df.groupby(group_by)['price'].quantile(q=0.75).reset_index().transform('price')

    return results_grouped


def get_prices_statistics_dict(cluster_prices, baseline=True):
    '''
        Get item prices statistics such as mean, median, max, min, first
        quantile, etc.

        cluster_prices (dict): item prices for each group (group_id -> list of
                               item ids).
        baseline (bool): if the first token grouping was used to generete the
                         groups.
    '''

    cluster_prices_statistics = {}

    for cluster, prices in cluster_prices.items():
        if baseline and '_' not in cluster:
            prices_statistics = {
                'mean': -1,
                'median': -1,
                'var': -1,
                'std': -1,
            }
        elif not baseline and cluster == '-1':
            prices_statistics = {
                'mean': -1,
                'median': -1,
                'var': -1,
                'std': -1,
            }
        else:
            prices_statistics = {
                'mean': np.mean(prices),
                'median': np.median(prices),
                'var': np.var(prices),
                'std': np.std(prices)
            }
        cluster_prices_statistics[cluster] = prices_statistics

    return cluster_prices_statistics


def pricing(itemlist, cluster_items, cluster_prices, dsc_unidade=True, year=True,
            remove_outliers=True, sample=False, threshold=0.3, baseline=True):
    '''
        Get item prices statistics such as mean, median, max, min, first
        quantile, etc.

        itemlist (ItemList object): contains the items and informations about
                                    the dataset.
        cluster_items (dict): dictionary of groups (group_id -> list of item ids).
        cluster_prices (dict): item prices for each group (group_id -> list of
                               item ids).
        dsc_unidade (bool): if the field 'dsc_unidade_medida' should be used for
                            pricing.
        year (bool): if the field 'ano' should be used for pricing.
        remove_outliers (bool): if the outlier prices should be removed.
        sample (bool): if the items of each group should be sampled before pricing.
        threshold (float): all groups which have mean/std > threshold, have the
                           outliers removed.
        baseline (bool): if the first token grouping was used to generete the
                         groups.
    '''

    if remove_outliers:
        cluster_prices, cluster_items = remove_outlier_prices(itemlist,
                            cluster_items, cluster_prices, threshold, baseline)
    if sample:
        cluster_prices, cluster_items = sample_per_decile(itemlist,
                        clisters_items, cluster_prices, threshold, baseline)

    items_clusters_df = get_items_dataframe(itemlist, cluster_items, baseline)
    cluster_prices_statistics = get_prices_statistics_df(items_clusters_df,
                                                         dsc_unidade)
    cluster_prices_statistics = add_first_token_column(cluster_prices_statistics)
    cluster_prices_statistics = add_outlier_column(cluster_prices_statistics)

    if year:
        cluster_prices_statistics_year = get_prices_statistics_df(items_clusters_df,
                                                             dsc_unidade, year)
        cluster_prices_statistics_year = add_first_token_column(cluster_prices_statistics_year)
        cluster_prices_statistics_year = add_outlier_column(cluster_prices_statistics_year)
    else:
        cluster_prices_statistics_year = None

    return cluster_prices_statistics, cluster_prices_statistics_year, items_clusters_df


def get_reference_prices(results, cluster_prices_statistics, dsc_unidade=True,
                         year=True):
    '''
        Get reference prices for the items in the test set.

        results (dict): results for the test set.
        cluster_price_statistics (DataFrame): prices statistics for the train set.
        dsc_unidade (bool): if the field 'dsc_unidade_medida' should be used for
                            pricing.
        year (bool): if the field 'ano' should be used for pricing.
    '''

    if dsc_unidade and year:
        indexes = ['cluster', 'dsc_unidade_medida', 'ano']
    elif dsc_unidade:
        indexes = ['cluster', 'dsc_unidade_medida']
    elif year:
        indexes = ['cluster', 'ano']
    else:
        indexes = ['cluster']
    prices_dict = cluster_prices_statistics.set_index(indexes).to_dict('index')
    item_reference_price = {}

    for result in results:
        if dsc_unidade and year:
            group = (result['cluster'], result['dsc_unidade_medida'], result['ano'])
        elif dsc_unidade:
            group = (result['cluster'], result['dsc_unidade_medida'])
        elif year:
            group = (result['cluster'], result['ano'])
        else:
            group = result['cluster']
        if group not in prices_dict:
            statistics = {
                'mean': -1,
                'median': -1,
                'var': -1,
                'std': -1,
                'count': -1,
                'max' : -1,
                'min' : -1,
                'quantile_1': -1,
                'quantile_3': -1
            }
        else:
            statistics = prices_dict[group]
        result.update(statistics)
        item_reference_price[result['item_id']] = result

    items_test_df = pd.DataFrame.from_dict(item_reference_price, 'index')
    items_test_df = add_first_token_column(items_test_df)

    return items_test_df
