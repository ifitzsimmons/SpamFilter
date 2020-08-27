from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol

import argparse
import json
import sys
import re
from math import log

WORD_RE = re.compile(r"[\w']+")

stop_words = []
with open('stopwords.txt') as infile:
  stop_words = infile.read().splitlines()

probabilities = None
with open('spam_classifier_data.json', 'r') as infile:
  probabilities = json.load(infile)

probability_spam = probabilities['total_spam'] / probabilities['total_emails']
probability_ham = probabilities['total_ham'] / probabilities['total_emails']

class TestModel(MRJob):
  def configure_args(self):
    ''' Set command line args for PageRank

    --num-emails: int
      total number of emails in training set
    '''

    super(TestModel, self).configure_args()
    self.add_passthru_arg(
      '--num-emails', type=int, required=True,
      help="total number of emails in training set"
    )

  def mapper(self, _, row):
    row_list = row.split(',')
    spam_classifier = row_list[0]
    text = ' '.join(row_list[1:])

    p_spam = 0
    p_ham = 0

    for word in set(WORD_RE.findall(text)):
      if word.lower() in probabilities:
        p_spam += probabilities[word.lower()]['spam_prob']
        p_ham += probabilities[word.lower()]['ham_prob']

    p_ham += log(probability_ham)
    p_spam += log(probability_spam)

    classifier = 'spam' if p_spam > p_ham else 'ham'

    if classifier == spam_classifier:
      yield 'correct', 1



  def reducer(self, accuracy, counts):
    count = sum(counts)

    model_acc = count / self.options.num_emails

    yield 'accuracy', model_acc

if __name__ == "__main__":
  TestModel.run()