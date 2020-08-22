from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol

import re

WORD_RE = re.compile(r"[\w']+")

stop_words = []
with open('stopwords.txt') as infile:
  stop_words = infile.read().splitlines()

class BuildSpamModel(MRJob):
  def mapper(self, _, row):
    row_list = row.split(',')
    spam_classifier = row_list[0]
    text = ' '.join(row_list[1:])

    if spam_classifier.lower() == 'spam':
      self.increment_counter('mapper', 'spam', 1)
    else:
      self.increment_counter('mapper', 'ham', 1)

    for word in set(WORD_RE.findall(text)):
      if word.lower() in stop_words or re.match('^\d+$', word):
        continue

      if spam_classifier.lower() == 'spam':
        yield word.lower(), (1, 0)
      else:
        yield word.lower(), (0,1)


  def reducer(self, word, counts):
    self.increment_counter('mapper', 'words', 1)
    spam_count = 0
    ham_count = 0

    for spam, ham in counts:
      if spam:
        spam_count += 1
      else:
        ham_count += 1

    yield word, (spam_count, ham_count)

if __name__ == "__main__":
  BuildSpamModel.run()