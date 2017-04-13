
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


class EmailClassifier:
    """ 
    An email classifier object.
    """

    def __init__(self):
        self.clf = MultinomialNB()
        self.vectorizer = CountVectorizer(max_df=.5, stop_words='english')
        self.scorer = make_scorer(accuracy_score)

    def train(self, emails, labels):
        """
        :param emails: A list of emails bodies with the subject being the first line. 
        :param labels: A list where each entry represents the correct label for the respective email body in the 
        emails parameter 
        """
        x = self.vectorizer.fit_transform(emails, labels)
        y = labels

        params = {
            'alpha': (0, .25, .5, .75, 1.),
            'fit_prior': (True, False),
        }

        grid = GridSearchCV(self.clf, params, scoring=self.scorer)
        grid.fit(x, y)

        self.clf = grid.best_estimator_

    def classify(self, emails):
        """
        :param emails: A list of emails bodies with the subject being the first line.
        :return: A list of labels as such as the first position is the label for the first emails position and so on.
        """
        x = self.vectorizer.transform(emails)
        return self.clf.predict(x)

    def save_to_file(self, file_name):
        """
        Saves the trained classifier to a file. 
        :param file_name: 
        :return: 
        """
        joblib.dump({'clf': self.clf, 'vec': self.vectorizer}, file_name)

    def load_from_file(self, file_name):
        """
        Load the trained classifier from given file.
        :param file_name: 
        :return: 
        """
        objs = joblib.load(file_name)
        self.clf = objs['clf']
        self.vectorizer = objs['vec']
