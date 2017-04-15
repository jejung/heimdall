from __future__ import print_function

from plugins import gmail
from learning.models import EmailClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv
import os
import pandas


def store_messages(force_search=False):
    if force_search or not os.path.isfile('data/gmail_samples.csv'):
        with open('data/gmail_samples.csv', 'wb') as data_file:
            writer = csv.DictWriter(data_file, fieldnames=['body', 'label'])
            writer.writeheader()
            samples = 0
            total_samples = 5000
            for message in gmail.get_emails(total_samples):
                print('{}/{}'.format(samples, total_samples))
                writer.writerow(message)
                samples += 1

if __name__ == '__main__':
    store_messages(False)
    messages_df = pandas.read_csv('data/gmail_samples.csv', encoding='utf-8')
    messages = messages_df['body'].values.astype('U')
    labels = messages_df['label'].values.astype('U')

    x_train, x_test, y_train, y_test = train_test_split(messages, labels, test_size=.20, random_state=42)

    clf = EmailClassifier()
    clf.train(x_train, y_train)
    predicted = clf.classify(x_test)

    print(accuracy_score(y_test, predicted))
