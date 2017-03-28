from __future__ import print_function

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups

train_data = fetch_20newsgroups(subset='train')

print('Size of train data:', len(train_data.data))
print('Target size:', len(train_data.target))


def count_aggregate_targets():
    result = {}
    for target in train_data.target:
        result[target] = result.get(target, 0) + 1

    return result


def plot_categories_dist():
    target_count = count_aggregate_targets()
    ind = np.arange(len(target_count.values()))
    fig, ax = plt.subplots()
    xaxis = ax.bar(ind, target_count.values(), 0.35, 0.35)
    ax.set_ylabel('Count')
    ax.set_title('Distribution of targets')
    ax.set_xticks(ind)
    ax.set_xticklabels(target_count.keys())

    ax.legend(xaxis, map(lambda x: str(x) + ' - ' + train_data.target_names[x], target_count.keys()))
    plt.show()


if __name__ == '__main__':
    plot_categories_dist()
