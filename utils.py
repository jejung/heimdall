import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


class Result:
    def __init__(self):
        self.pre_processing_time = 0
        self.training_time = 0
        self.testing_time = 0
        self.score = 0
        self.clf = None
        self.vectorizer = None
        self.name = None

    def get_name(self):
        return self.name if self.name else self.vectorizer.__class__.__name__ + '/' + self.clf.__class__.__name__


def benchmark(
    vectorizer, clf, data_train, data_test, with_pca=False, pca_class=TruncatedSVD,
    unified_times=True
):
    result = Result()
    result.vectorizer = vectorizer
    result.clf = clf

    train_size_mb = size_mb(data_train.data)
    test_size_mb = size_mb(data_test.data)

    print('Using {} vectorizer for a {} model'.format(vectorizer.__class__.__name__, clf.__class__.__name__))
    print('Pre-processing')
    t0 = time()
    X_train = vectorizer.fit_transform(data_train.data)
    y_train = data_train.target
    if unified_times:
        result.training_time += time() - t0
        t0 = time()
    X_test = vectorizer.transform(data_test.data)
    y_test = data_test.target
    if unified_times:
        result.testing_time += time() - t0
    if with_pca:
        svd = pca_class(algorithm='arpack')
        X_train = svd.fit_transform(X_train, y_train)
        X_test = svd.transform(X_test)
    if not unified_times:
        duration = time() - t0
        result.pre_processing_time = duration
    else:
        result.pre_processing_time = result.training_time + result.testing_time
    print('Done in {:.3f}s at {:.3f} MB/s'.format(
        result.pre_processing_time,
        (train_size_mb + test_size_mb) / result.pre_processing_time)
    )

    print('Training')
    t0 = time()
    clf.fit(X_train, y_train)
    duration = time() - t0
    result.training_time += duration
    print('Done in {:.3f}s at {:.3f} MB/s'.format(duration, train_size_mb / duration))

    print('Testing')
    t0 = time()
    prediction = clf.predict(X_test)
    duration = time() - t0
    print('Done in {:.3f}s at {:.3f} MB/s'.format(duration, test_size_mb / duration))
    result.testing_time += duration

    result.score = accuracy_score(y_test, prediction)
    print('Accuracy score of {:.3f}'.format(result.score))

    return result


def plot_results(results, image_name, title="Benchmark"):
    results = [(x.get_name(), x.score, x.training_time, x.testing_time) for x in results]

    indices = np.arange(len(results))

    clf_names, score, training_time, test_time = [[x[i] for x in results] for i in range(4)]
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 10))
    plt.title(title)
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
