from time import time
from utils import size_mb, benchmark, plot_results, Result
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from learning.models import EmailClassifier
from learning.stop_words import ENGLISH
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score


def get_final_solution_results(data_train, data_test):
    email_clf = EmailClassifier(stop_words=ENGLISH)
    result = Result()
    result.name = 'Final version'
    result.vectorizer = email_clf.vectorizer
    result.clf = email_clf.clf

    print('Final version -----------------')
    print('Training')
    t0 = time()
    email_clf.train(data_train.data, data_train.target)
    result.training_time = time() - t0
    print('Done in {:.3f}'.format(result.training_time))

    print('Training')
    t0 = time()
    predicted = email_clf.classify(data_test.data)
    result.testing_time = time() - t0
    print('Done in {:.3f}'.format(result.testing_time))

    result.score = accuracy_score(data_test.target, predicted)

    return result

if __name__ == '__main__':
    categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', ]

    print("Loading 20 newsgroups dataset for categories:")
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    print('data loaded')

    train_size_mb = size_mb(data_train.data)
    test_size_mb = size_mb(data_test.data)

    print('Train data size: {:.3}MB'.format(train_size_mb))
    print('Test data size: {:.3}MB'.format(test_size_mb))

    results = []

    count = CountVectorizer()
    clf = MultinomialNB()
    results.append(benchmark(vectorizer=count, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))
    clf = KNeighborsClassifier()
    results.append(benchmark(vectorizer=count, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))
    clf = LinearSVC()
    results.append(benchmark(vectorizer=count, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))

    tfid = TfidfVectorizer()
    clf = MultinomialNB()
    results.append(benchmark(vectorizer=tfid, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))
    clf = KNeighborsClassifier()
    results.append(benchmark(vectorizer=tfid, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))
    clf = LinearSVC()
    results.append(benchmark(vectorizer=tfid, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))

    hash = HashingVectorizer(non_negative=True)
    clf = MultinomialNB()
    results.append(benchmark(vectorizer=hash, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))
    clf = KNeighborsClassifier()
    results.append(benchmark(vectorizer=hash, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))
    clf = LinearSVC()
    results.append(benchmark(vectorizer=hash, clf=clf, data_train=data_train, data_test=data_test, unified_times=True))

    results.append(get_final_solution_results(data_train, data_test))

    plot_results(results, image_name='data/comparision.png', title='Comparision')
