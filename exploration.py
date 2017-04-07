from __future__ import print_function

import logging
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF


def count_aggregate_targets(train_data):
    print('Size of train data:', len(train_data.data))
    print('Target size:', len(train_data.target))

    result = {}
    for target in train_data.target:
        result[target] = result.get(target, 0) + 1

    return result


def plot_categories_dist():
    train_data = load_filtered_data('train')
    target_count = count_aggregate_targets(train_data)
    ind = np.arange(len(target_count.values()))
    fig, ax = plt.subplots()
    xaxis = ax.bar(ind, target_count.values(), 0.35, 0.35)
    ax.set_ylabel('Count')
    ax.set_title('Distribution of targets')
    ax.set_xticks(ind)
    ax.set_xticklabels(target_count.keys())

    ax.legend(xaxis, map(lambda x: str(x) + ' - ' + train_data.target_names[x], target_count.keys()))
    plt.show()


def load_filtered_data(subset):
    return fetch_20newsgroups(subset=subset, categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', ], )


def print_top_words(topic_idx, model, feature_names, n_top_words):
    plot = 321 + topic_idx

    ax = plt.subplot(plot)
    topic = model.components_[topic_idx]
    ind = np.array(range(n_top_words))
    ax.bar(ind, [topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]], 0.35, 0.35)
    ax.set_ylabel('Score')
    ax.set_title('Topic {}'.format(topic_idx + 1))
    ax.set_xticks(ind)
    ax.set_xticklabels([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]], rotation='vertical')
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))


def analyze_vectorizer(data, document_count, vectorizer):
    feature_matrix = vectorizer.fit_transform(data.data)

    nmf = NMF(n_components=document_count, random_state=1, alpha=.1, l1_ratio=.5).fit(feature_matrix)
    count_feature_names = vectorizer.get_feature_names()

    plt.figure(1)
    for i in range(document_count):
        print_top_words(i, nmf, count_feature_names, 20)

    plt.show()

if __name__ == '__main__':
    # plot_categories_dist()
    data_train = load_filtered_data('train')
    filtered_test_data = load_filtered_data('test')
    print('Size of train data:', len(data_train.data))
    print('Size of test data:', len(filtered_test_data.data))

    # split a training set and a test set
    y_train, y_test = data_train.target, filtered_test_data.target

    topic_count = 6
    analyze_vectorizer(data_train, topic_count, CountVectorizer())
    analyze_vectorizer(data_train, topic_count, TfidfVectorizer())
