# Spam Classifier

* [Pre-Processing](#pre-processing)
* [Naive Bayes Theorem](#naive-bayes)
* [MapReduce Implementation](#mapreduce-instructions)
  * [Math](#math)
  * [Training](#training-the-model)
  * [Testing](#testing-the-model)
  * [How To Execute](#execution)
  * [Results](#mapreduce-classifier-results)

This repository contains implementations of Spam Classifiers using **Naive Bayes** with implementations written with both MapReduce (`MRJob`) and Spark (`pyspark`).

## Pre-Processing
Please download the following dataset and store it in the root directory of this repository:
https://www.kaggle.com/uciml/sms-spam-collection-dataset

Once the data has been downloaded and extracted please ensure the name of the dataset is **`datasets_483_982_spam.csv`**

## Naive Bayes
The probability that an email is spam given a number of words relies on **Naive Baye's Theorem (1.0)**. Naive Baye's assumes that all observables are independent of one another, meaning that the appearance of one word in the text does not affect the likelihood of another word appearing in the text, which is why it is *naive*.
```math
P(\vec{W}|S) = \prod_{w \epsilon \vec{W}}{p(w|S)} \tag{1.0}
```
In **Equation 1.0**, $\vec{W}$ represents a list of words. In plain english, this equation states that the probability of a condition (list of words) given another condition ("the email is spam") is the product of the probability each words existance given that the email is spam.

Combining **Equation 1.0** with Baye's Theorem (**Equation 2.0**), we can calculate the probability that an email is spam given a list of words:
```math
P(S|\vec{W}) = \frac{P(\vec{W}|S)P(S)}{P(\vec{W}|S)P(S) + P(\vec{W}|\neg{S})P(\neg{S)}} \tag{2.0}
```

## MapReduce Instructions

Classification in [MapReduce](./MapReduceSpam) must be done in several steps. First, you must **Train** the model, then, you may **Test** the model.

For this implementation, we decided on a `80/20` train/test split, so bear in mind that 80% of the data you provide will be used for training the model.

Also note that this model is set up for **.csv data only**.

### Math
#### TODO - clean up math section
Add parts about Laplace smoothing and underflow

### Training the Model
The `Driver.py` file is responsible for quite a few things. The first thing it does is take the initial data and creates **two** files: `training.csv` and `test.csv`.

It then runs `TrainModel.py` by passing in the input (`training.csv`) and a few other required arguments. The following block explains the command line arguments for `TrainModel.py` along with a brief description of each argument.
```python
'''
  --total-spam: int
    total number of spam emails

  --total-ham: int
    total number of ham emails

  --total-words: int
    total number of unique words in training set
'''
```

The Training Model uses *Laplace Smoothing* to dampen the effects of words that do not exist and otherwise have a probability of 0.

After the model has been trained, a file is created (`spam_classifier_data.json`) with all of the words that appeared in the training set as keys and the probability that the word is *spam* or *ham* as values.

`spam_classifier_data.json` is then used to test the model.

\**Please note that the training algorithm omits all stop words as defined by the `nltk` library and also omits and numeric strings.

### Testing The Model
In order to *test* the model, you must have a `.csv` file in the same format as the *training* data.

Testing is handled by `TestModel.py`, which eventually emits the accuracy of the model as a percentage. This percentage is the number of correct predictions divided by the number of samples in the test data set. As such, it is important to pass the number of test samples to the `MapReduce` job via the command line argument `--num-emails`. The `--num-emails` argument specifies the number of emails in the test dataset.

### Execution
To train the model, you need only to run the driver with a given dataset, solong as the dataset is in `.csv` format. In the following example, the dataset is stored in the root directory of the project, but note that it can be stored anywhere

```bash
$ python Driver.py --input-file ../datasets_483_982_spam.csv
No configs specified for inline runner
No configs specified for inline runner

Training Data: ./training.csv
Test Data: ./test.csv
```

Then, the only thing left to do is test the accuracy of the model by running `TestModel.py` with your test dataset.

```bash
No configs found; falling back on auto-configuration
No configs specified for inline runner
Creating temp directory /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153
Running step 1 of 1...
job output is in /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153/output
Streaming final output from /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153/output...
"accuracy"	0.9820466786355476
Removing temp directory /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153...
```

### MapReduce Classifier Results
In this case, the MapReduce Spam Classifier was **98.2%** accurate.
______

## PySpark Instructions

### Setup

____
## Results

### PySpark Results

### MapReduce Results

### Conclusions