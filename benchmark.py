import matplotlib.pyplot as plt
import numpy as np
import sys

from utils import size_mb
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', ]

print("Loading 20 newsgroups dataset for categories:")
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
print('data loaded')

train_size_mb = size_mb(data_train.data)
test_size_mb = size_mb(data_test.data)

print('Train data size: {:.3}MB'.format(train_size_mb))
print('Test data size: {:.3}MB'.format(test_size_mb))

target_names = data_train.target_names

y_train, y_test = data_train.target, data_test.target


class Result:
    def __init__(self):
        self.pre_processing_time = 0
        self.training_time = 0
        self.testing_time = 0
        self.score = 0
        self.clf = None
        self.vectorizer = None


def benchmark(vectorizer, clf, with_pca=False):
    result = Result()
    result.vectorizer = vectorizer
    result.clf = clf

    print('Using {} vectorizer for a {} model'.format(vectorizer.__class__.__name__, clf.__class__.__name__))
    print('Pre-processing')
    t0 = time()
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
    if with_pca:
        svd = TruncatedSVD(algorithm='arpack')
        X_train = svd.fit_transform(X_train, y_train)
        X_test = svd.transform(X_test)
    duration = time() - t0
    result.pre_processing_time = duration
    print('Done in {:.3f}s at {:.3f} MB/s'.format(duration, (train_size_mb + test_size_mb) / duration))

    print('Training')
    t0 = time()
    clf.fit(X_train, y_train)
    duration = time() - t0
    result.training_time = duration
    print('Done in {:.3f}s at {:.3f} MB/s'.format(duration, train_size_mb / duration))

    print('Testing')
    t0 = time()
    prediction = clf.predict(X_test)
    duration = time() - t0
    print('Done in {:.3f}s at {:.3f} MB/s'.format(duration, test_size_mb / duration))
    result.testing_time = duration

    result.score = accuracy_score(y_test, prediction)
    print('Accuracy score of {:.3f}'.format(result.score))

    return result


def plot_results(results, image_name):
    results = [
        (x.vectorizer.__class__.__name__ + '/' + x.clf.__class__.__name__, x.score, x.training_time, x.testing_time) for
        x in results]

    indices = np.arange(len(results))

    clf_names, score, training_time, test_time = [[x[i] for x in results] for i in range(4)]
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 10))
    plt.title("Benchmark")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time", color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.37, i, c)

    plt.savefig(image_name)
    plt.show()


mode = sys.argv[1] if len(sys.argv) > 1 else 'benchmark'

if mode == 'benchmark':
    results = []

    hashing = HashingVectorizer(non_negative=True)
    knn = KNeighborsClassifier()
    results.append(benchmark(hashing, knn))
    svc = LinearSVC()
    results.append(benchmark(hashing, svc))
    naive_bayes = MultinomialNB()
    results.append(benchmark(hashing, naive_bayes))

    count = CountVectorizer()
    knn = KNeighborsClassifier()
    results.append(benchmark(count, knn))
    svc = LinearSVC()
    results.append(benchmark(count, svc))
    naive_bayes = MultinomialNB()
    results.append(benchmark(count, naive_bayes))

    tfidf = TfidfVectorizer()
    knn = KNeighborsClassifier()
    results.append(benchmark(tfidf, knn))
    svc = LinearSVC()
    results.append(benchmark(tfidf, svc))
    naive_bayes = MultinomialNB()
    results.append(benchmark(tfidf, naive_bayes))
    image_name = 'data/benchmark.png'

    plot_results(results, image_name)

elif mode == 'datapreprocessing':

    results = []

    count = CountVectorizer(max_features=100)
    knn = KNeighborsClassifier()
    results.append(benchmark(count, knn, with_pca=True))
    svc = LinearSVC()
    results.append(benchmark(count, svc, with_pca=True))
    naive_bayes = MultinomialNB()
    results.append(benchmark(count, naive_bayes, with_pca=True))

    tfidf = TfidfVectorizer(max_features=100)
    knn = KNeighborsClassifier()
    results.append(benchmark(tfidf, knn, with_pca=True))
    svc = LinearSVC()
    results.append(benchmark(tfidf, svc, with_pca=True))
    naive_bayes = MultinomialNB()
    results.append(benchmark(tfidf, naive_bayes, with_pca=True))

    image_name = 'data/preprocessing.png'

    plot_results(results, image_name)
