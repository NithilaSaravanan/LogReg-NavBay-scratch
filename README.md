# Implementing Logistic Regression and Naive Bayes from scratch
The primary motive of this project is to understand the crux of two major classification algorithms - Logistic Regression
and Naive Bayes. These two algorithms were employed on four different UCI datasets.
- It is important to pre-process data in order to get accurate
results
- Small alpha will not assist in attaining the optimum but
large alpha will overshoot it
- K Fold CV is useful in figuring out the hyperparameters,
thus avoiding errors
- Logistic Regression outperforms Naive Bayes except
when the model ran on breasy cancer data - this is true
because of the scarcity of training data

### Abstract from the report

In this project, the performance of two Machine Learning (ML) algorithms, namely Logistic Regression (LR) and
Gaussian Naive Bayes(GNB) was investigated on four benchmark
datasets. The data was preprocessed where missing or irrelevant
variables were removed and certain datasets were normalized
accordingly. Furthermore, all the datasets were split into training,
testing and validation sets. The models were implemented and Kfold
cross validation was used. Moreover, some additional features
were proposed for means of improving the performance, which
was chosen to be the log, quadratic and interactive terms of the
most discriminant features, these were added only to GNB.
