import pandas as pd
import numpy as np
import collections


def get_group_name(subgroups):
    first_token = str(subgroups[0].split('_')[0])
    ids = ""
    for group in subgroups:
        id_ = str(group.split('_')[1])
        ids += "_" + id_

    return first_token + ids


def select_items(items_df, clusters_df):

    # Adiciona a informação de grupo e ruído no dataframe de itens
    items_df = items_df.merge(clusters_df, left_on='item_id', right_on='item_id')

    # Exclui grupos de ruídos e grupos não particionados
    items_df = items_df[items_df.item_ruido == 0]
    items_df['original_desc'] = items_df['original_prep'].apply(lambda x: ' '.join(eval(x)))

    # Quantidade de descrições por grupo
    groups_count_df = items_df.groupby('grupo', as_index=True).count().sort_values('item_id', ascending=False)['original_desc']
    print(f'Quantidade de grupos considerados: {len(groups_count_df)}')

    items_df.set_index('item_id', inplace=True)
    groups_count = groups_count_df.reset_index()

    groups = groups_count.values.tolist()

    return items_df, groups


def regrouping(description_count):

    desc_canon_groups = collections.defaultdict(list)

    for desc, subgroups in description_count.items():
        subgroups.sort()
        group_name = get_group_name(subgroups)
        group_info = {'description': desc,
                 'groups': subgroups
                }
        desc_canon_groups[group_name] = group_info

    return desc_canon_groups


def get_clusters_items(final_clusters, clusters):

    clusters_items = {}

    for group_name, group_info in final_clusters.items():
        items = []
        for subgroup in group_info['groups']:
            items += clusters[subgroup]
        clusters_items[group_name] = items

    return clusters_items
