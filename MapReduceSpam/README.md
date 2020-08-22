# Spam Classifier using MRJob (MapReduce)

This repository will generate a json document, `spam_data.json`, that can be used to calculate the probability that an email is spam given a list of words in the email.

##Using the Driver
The `Driver.py` file will take an input and generate the necessary spam classifier data in an output file, `spam_data.json`. It does so by calling the `BuildSpamModel` class, which extends `MRJob`.

### Input
`Driver.py` and `BuildSpamModel.py` both expect a comma separated file with **two** columns, where the first item is either `"spam"` or `"ham"` and the second item is the text contained in the email. This data is used to count the number of times a given word occur in spam and ham emails.

###Training

`BuildSpamModel.py` emits a words and word information as key value pairs. It also counts the number of *spam* emails and the number of *ham* emails. The Driver takes all of this data and produces a `json` object that contains information about each word and the dataset as a whole. The structure of `sample_data.json` is as follows:
```json
{
  "total_spam": 747,
  "total_ham": 4826,
  "total_emails": 5573,
  "<word1>": {
    "spam_count": 23,
    "ham_count": 342
  },
  "<word2>": {
    "spam_count": 21,
    "ham_count": 0
  },
  .
  .
  .
}
```
\**Please note that the training algorithm omits all stop words as defined by the `nltk` library and also omits and numeric strings.

### Execution
To run the driver and generate the spam classifier data, execute the following command from the command line:
```bash
$ python Driver.py --input-file ~/Downloads/datasets_483_982_spam.csv
```

Note that any CSV with the format specified in the **Inputs** section will work.

______

##Calculating Probability
In order to calculate the probability that an email is spam given a selection of words' existence in the email, you **must** run the driver to gereate the training data. `CalculateProbability.py` is dependent on a valid `spam_data.json` file.

###Math
The probability that an email is spam given a number of words relies on **Naive Baye's Theorem (1.0)**. Naive Baye's assumes that all observables are independent of one another, meaning that the appearance of one word in the text does not affect the likelihood of another word appearing in the text, which is why it is *naive*.
```math
P(\vec{W}|S) = \prod_{w \epsilon \vec{W}}{p(w|S)} \tag{1.0}
```
In **Equation 1.0**, $\vec{W}$ represents a list of words. In plain english, this equation states that the probability of a condition (list of words) given another condition ("the email is spam") is the product of the probability each words existance given that the email is spam.

Combining **Equation 1.0** with Baye's Theorem (**Equation 2.0**), we can calculate the probability that an email is spam given a list of words:
```math
P(S|\vec{W}) = \frac{P(\vec{W}|S)P(S)}{P(\vec{W}|S)P(S) + P(\vec{W}|\neg{S})P(\neg{S)}}
```


###Execution
The input for `CalculateProbability.py` is a **space separated** list of of words. The words provided will be used to calculate the probability that an email is spam if it contains *all* of the words provided.

#### Examples
```bash
$ python CalculateProbability.py account play
The probability that an email is spam if it contains the word(s) "account, play" is 87.52%
```

```bash
$ python CalculateProbability.py theory
The probability that an email is spam if it contains the word(s) "theory" is 0.00%
```

```bash
$ python CalculateProbability.py dsfgasdf
"dsfgasdf" does not exist in the corpora, ProbabilitySpam = 0%
```