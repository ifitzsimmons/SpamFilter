# Spam Classifier

* [Pre-Processing](#pre-processing)
* [Naive Bayes Theorem](#naive-bayes)
* [MapReduce Implementation](#mapreduce-instructions)
  * [Math](#math)
  * [Training](#training-the-model)
  * [Testing](#testing-the-model)
  * [How To Execute](#execution)
  * [Results](#mapreduce-classifier-results)
* [Spark Implementation](#pyspark-instructions)
  * [Setup](#setup)
    * [Dependencies](#dependencies)
  * [Results](#pyspark-results)
* [Conclusions](#conclusions)

This repository contains implementations of Spam Classifiers using **Naive Bayes** with implementations written with both MapReduce (`MRJob`) and Spark (`pyspark`).

## Pre-Processing
Please download the following dataset and store it in the root directory of this repository:
https://www.kaggle.com/uciml/sms-spam-collection-dataset

Once the data has been downloaded and extracted please ensure the name of the dataset is **`datasets_483_982_spam.csv`**

## Naive Bayes
The probability that an email is spam given a number of words relies on **Naive Baye's Theorem (1.0)**. Naive Baye's assumes that all observables are independent of one another, meaning that the appearance of one word in the text does not affect the likelihood of another word appearing in the text, which is why it is *naive*.

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(\vec{W}|S)=\prod_{w \epsilon \vec{W}}{p(w|S)}" title="Naive Bayes" />
</div>

In **Equation 1.0**, $\vec{W}$ represents a list of words. In plain english, this equation states that the probability of a condition (list of words) given another condition ("the email is spam") is the product of the probability each words existance given that the email is spam.

Combining **Equation 1.0** with Baye's Theorem (**Equation 2.0**), we can calculate the probability that an email is spam given a list of words:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(S|\vec{W})=\frac{P(\vec{W}|S)P(S)}{P(\vec{W})}" title="Bayes Theorem" />
</div>

## MapReduce Instructions

Classification in [MapReduce](./MapReduceSpam) must be done in several steps. First, you must **Train** the model, then, you may **Test** the model.

For this implementation, we decided on a `80/20` train/test split, so bear in mind that 80% of the data you provide will be used for training the model.

Also note that this model is set up for **.csv data only**.

### Math
To calculate the probabaility that an email is **_not_** spam, the **Equation 2.0** becomes:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(\neg{S}|\vec{W})=\frac{P(\vec{W}|\neg{S})P(\neg{S})}{P(\vec{W})}" title="Bayes Theorem" />
</div>

So, when comparing $ P(S|\vec{W}) $ to $ P(\neg{S}|\vec{W}) $ we can ignore the common denominator and say that the email can be classified by the following:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\&space;P(S|\vec{W})=P(\vec{W}|S)P(S)" title="Classification" /><p>and</p>
<img src="https://latex.codecogs.com/svg.latex?\&space;P(\neg{S}|\vec{W})=P(\vec{W}|\neg{S})P(\neg{S})" title="Classification" /><p>SO</p>
<img src="https://latex.codecogs.com/svg.latex?\&space;c_{nb}=argmax_c(P(S|\vec{W}), P(\neg{S}|\vec{W})) " title="Classification" />
</div>

### Smoothing
In the classification equation represented above, the model is strongly affected by zero probabilities. Any word that occurs in the test data does not appear in the training set will cause a prediciton of 0% spam **and** ham. To account for 0 probabilities, the training model factors in $ Laplace $ $ Smoothing $. For each word, the probability that an email is spam given a specific word, $ w $ is as follows:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(S|w)=\frac{count(w, S = True) + 1}{count_{docs}(S = True) + |\vec{W}|}" title="Laplace Smoothing" />
</div>

In the above equation, $ count(w, S = True) $ is the number of times the word, $ w $ occured in a spam email, $ count_{docs}(S = True) $ is the total number of spam emails, and $ |\vec{W}| $ is the total number of words in the training set.

### Underflow
Given the above equation, the probabilities of certain words can be extremely small, and since we will be using $ Naive $ $ Bayes $ to multiply these probabilies, the multiplication may cause underflow. To avoid underflow, we take advantage of logarithmic property:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\log{xy}=log{x}+\log{y}" title="Laplace Smoothing" />
</div>

By adding the $ \log $ of the probabilities, we avoid underflow due to multiplication of small numbers.

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
$ python TestModel.py test.csv --num-emails 1671
```

\** where `num-emails` can be found by running `$ wc -l test.csv` from the command line.

```bash
No configs found; falling back on auto-configuration
No configs specified for inline runner
Creating temp directory /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153
Running step 1 of 1...
job output is in /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153/output
Streaming final output from /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153/output...
"accuracy"	0.9814482345900658
Removing temp directory /var/folders/93/nz00qrh11077bm74kcjw5pyr0000gn/T/TestModel.mimian.20200822.222431.591153...
```

### MapReduce Classifier Results
In this case, the MapReduce Spam Classifier was **98.1%** accurate.
___

## PySpark Instructions
The `pyspark` implementation is much simpler to run thanks to `MLlib`'s `NaiveBayes` model. In terms of configuration, you must only ensure that your environment has been set up to run `pyspark` applications.

### Setup
To configure the Spam Classifier implemented with `Spark`, you must take the first block of code and point the `findspark.init` to the location on your machine where `pyspark` is installed.
```diff
import findspark
- findspark.init('/home/ubuntu/spark-3.0.0-bin-hadoop2.7')
+ findspark.init('path/to/spark')
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spam').getOrCreate()
```

In addition, be mindful of where the model is reading the CSV from, which is the top level of this directory. If you're data lives outside of this directory, make the following change:
```diff
- data = spark.read.csv('../datasets_483_982_spam.csv', inferSchema=True, header=True)
+ data = spark.read.csv('path/to/datasets_483_982_spam.csv', inferSchema=True, header=True)
```

#### Dependencies
In order to run the Spam Classifier with `pyspark`, you must have the following dependencies installed on your machine:
* `findspark`
* `py4j`
* `pyspark`

### PySpark Results
Pyspark gives the user the ability to perform a random train/test split on the data. Because of this, the reported accuracy of the model changes each time the model is run.

On average, the Spam Classifier implemented in `pyspark` reported an accuracy of about **97-98%**.
___

## Conclusions
The manual implementation of Naive Bayes in *MapReduce* and the Naive Bayes model implemented by *PySpark*\`s `MLlib` produce very similar results, with a difference in accuracy of less than **1%**.

While *MapReduce* produces a slightly more accurate model, it requires much more overhead to set up and has some defficiencies.

First, it is difficult to create a truly random train/test split on the input file with *MapReduce*. The way we accomplished splitting the data was by writing the first 70% of lines to the train file, and the remaining 30% to the test file. This makes the *MapReduce* result repeatable, but leaves the model more susceptible to training problems.

The second issue is that *MapReduce*, by comparison to *Spark*, takes longer to run. Not only is the combined run time longer for *MapReduce*, but it is also more difficult for the user to interact with and understand.

Since *Spark* loads all data into memory, *Spark* is able to process this dataset much faster than *MapReduce*, which is reading the data off of disk.

