import sys

from utils import size_mb, benchmark, plot_results
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import LinearSVC

categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', ]

print("Loading 20 newsgroups dataset for categories:")
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
print('data loaded')

train_size_mb = size_mb(data_train.data)
test_size_mb = size_mb(data_test.data)

print('Train data size: {:.3}MB'.format(train_size_mb))
print('Test data size: {:.3}MB'.format(test_size_mb))

mode = sys.argv[1] if len(sys.argv) > 1 else 'benchmark'

if mode == 'benchmark':
    results = []

    hashing = HashingVectorizer(non_negative=True)
    knn = KNeighborsClassifier()
    results.append(benchmark(hashing, knn, data_train, data_test))
    svc = LinearSVC()
    results.append(benchmark(hashing, svc, data_train, data_test))
    naive_bayes = MultinomialNB()
    results.append(benchmark(hashing, naive_bayes, data_train, data_test))

    count = CountVectorizer()
    knn = KNeighborsClassifier()
    results.append(benchmark(count, knn, data_train, data_test))
    svc = LinearSVC()
    results.append(benchmark(count, svc, data_train, data_test))
    naive_bayes = MultinomialNB()
    results.append(benchmark(count, naive_bayes, data_train, data_test))

    tfidf = TfidfVectorizer()
    knn = KNeighborsClassifier()
    results.append(benchmark(tfidf, knn, data_train, data_test))
    svc = LinearSVC()
    results.append(benchmark(tfidf, svc, data_train, data_test))
    naive_bayes = MultinomialNB()
    results.append(benchmark(tfidf, naive_bayes, data_train, data_test))
    image_name = 'data/benchmark.png'

    plot_results(results, image_name)

elif mode == 'datapreprocessing':

    results = []

    count = CountVectorizer(max_features=100)
    knn = KNeighborsClassifier()
    results.append(benchmark(count, knn, data_train, data_test, with_pca=True))
    svc = LinearSVC()
    results.append(benchmark(count, svc, data_train, data_test, with_pca=True))
    naive_bayes = MultinomialNB()
    results.append(benchmark(count, naive_bayes, data_train, data_test, with_pca=True))

    tfidf = TfidfVectorizer(max_features=100)
    knn = KNeighborsClassifier()
    results.append(benchmark(tfidf, knn, data_train, data_test, with_pca=True))
    svc = LinearSVC()
    results.append(benchmark(tfidf, svc, data_train, data_test, with_pca=True))
    naive_bayes = MultinomialNB()
    results.append(benchmark(tfidf, naive_bayes, data_train, data_test, with_pca=True))

    image_name = 'data/preprocessing.png'

    plot_results(results, image_name)
