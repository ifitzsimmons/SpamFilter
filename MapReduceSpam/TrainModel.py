from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol

from math import log

class TrainModel(MRJob):
  '''
  Train Spam Classifier with Preprocessed Training Data
  '''

  INPUT_PROTOCOL = JSONProtocol

  def configure_args(self):
    ''' Set command line args for PageRank

    --total-spam: int
      total number of spam emails

    --total-ham: int
      total number of ham emails

    --total-words: int
      total number of unique words in training set
    '''

    super(TrainModel, self).configure_args()
    self.add_passthru_arg(
      '--total-spam', type=int, required=True,
      help="total number of spam emails in training set"
    )

    self.add_passthru_arg(
      '--total-ham', type=int, required=True,
      help="total number of ham emails in training set"
    )

    self.add_passthru_arg(
      '--total-words', type=int, required=True,
      help="total number of unique words in training set"
    )

  def mapper(self, word, counts):
    '''
    Calculate probability of spam and probability of
    ham for each word in training set
    '''
    spam_count, ham_count = counts

    prob_word_spam = (spam_count + 1) / (self.options.total_spam + self.options.total_words)
    prob_word_ham = (ham_count + 1) / (self.options.total_ham + self.options.total_words)

    prob_word_spam = log(prob_word_spam)
    prob_word_ham = log(prob_word_ham)

    yield word, (prob_word_spam, prob_word_ham)

if __name__ == "__main__":
  TrainModel.run()