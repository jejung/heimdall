from learning.models import EmailClassifier
from unittest import TestCase
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score


class EmailClassifierTest(TestCase):
    def setUp(self):
        self.categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', ]
        self.test_data = fetch_20newsgroups(subset='test', categories=self.categories, shuffle=True, random_state=42)
        self.train_data = fetch_20newsgroups(subset='train', categories=self.categories, shuffle=True, random_state=42)

    def test_assert_accuracy_over_threshold(self):
        """
        Asserts that the model accuracy is over the desired threshold
        :return: 
        """
        threshold = .94
        clf = EmailClassifier()
        clf.train(self.train_data.data, self.train_data.target)
        accuracy = self.get_clf_score(clf)
        self.assertGreaterEqual(accuracy, threshold)

    def test_assert_a_saved_classifiers_performs_the_same_when_loaded(self):
        first_clf = EmailClassifier()
        first_clf.train(self.train_data.data, self.train_data.target)
        first_clf.save_to_file('/tmp/heimdall_test_clf.pkl')
        first_score = self.get_clf_score(first_clf)
        second_clf = EmailClassifier()
        second_clf.load_from_file('/tmp/heimdall_test_clf.pkl')
        second_score = self.get_clf_score(second_clf)
        self.assertGreaterEqual(second_score, first_score)

    def get_clf_score(self, clf):
        pred = clf.classify(self.test_data.data)
        return accuracy_score(self.test_data.target, pred)
