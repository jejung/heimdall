from __future__ import print_function

from plugins import gmail
import csv

with open('data/gmail_samples.csv', 'wb') as data_file:
    writer = csv.DictWriter(data_file, fieldnames=['body', 'label'])
    writer.writeheader()
    samples = 0
    total_samples = 5000
    for message in gmail.get_emails(total_samples):
        print('{}/{}'.format(samples, total_samples))
        writer.writerow(message)
        samples += 1
