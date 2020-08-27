from BuildSpamModel import BuildSpamModel
from TrainModel import TrainModel
import argparse
import json
import sys

def add_arguments():
  '''Add command line argument for PageRank calculation string

  --input-file: str
    file of graph nodes

  '''

  parser.add_argument("--input-file", required=True,
          help=("File or files with node structure"))

def create_train_test(input_file):
  ''' This function creates test data and training data

  This function splits the input data with a 70/30, train/test split.

  PARAMETERS
  ----------
  input_file: str
    Name of the input file, assumes csv

  RETURNS
  -------
  train_file: str
    name of file that contains training data

  test_file: str
    name of file that contains test data
  '''

  test_file, train_file = ('./test.csv', './training.csv')

  with open(input_file, encoding='ISO-8859-1') as infile:
    lines = infile.readlines()
    total_lines = len(lines)

    with open(train_file, 'w') as training, open(test_file, 'w') as test:
      for line_num, line in enumerate(lines):
        if line_num / total_lines <= 0.7:
          training.write(line)
        else:
          test.write(line)

  return train_file, test_file,


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  add_arguments()
  args = parser.parse_args()

  infile = args.input_file

  train_file, test_file = create_train_test(infile)

  mr_job = BuildSpamModel([train_file, '--output-dir=output'])

  with mr_job.make_runner() as runner:
    runner.run()
    counters = runner.counters()

    training_data = {}

    spam_emails =counters[0]['mapper']['spam']
    ham_emails = counters[0]['mapper']['ham']
    total_words = counters[0]['mapper']['words']

    total_emails = spam_emails + ham_emails

    ''' Train the classifier with training data'''
    input_dir = 'output'
    total_spam_arg = f'--total-spam={spam_emails}'
    total_ham_arg = f'--total-ham={ham_emails}'
    total_words_arg = f'--total-words={total_words}'


    mr_job2 = TrainModel([input_dir, total_spam_arg, total_ham_arg, total_words_arg])
    with mr_job2.make_runner() as runner2:
      runner2.run()

      for word, v in mr_job.parse_output(runner2.cat_output()):
        spam_prob, ham_prob = v
        training_data[word] = {
          'spam_prob': spam_prob,
          'ham_prob': ham_prob
        }

      training_data['total_emails'] = total_emails
      training_data['total_spam'] = spam_emails
      training_data['total_ham'] = ham_emails

      with open('spam_classifier_data.json', 'w') as json_file:
        json.dump(training_data, json_file)

  out_string = f'\nTraining Data: {train_file}\nTest Data: {test_file}\n'
  print(out_string)
