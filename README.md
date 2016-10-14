# Mini Project 2

## Abstract Prediction Text Classification Problem

Assignment for Comp 551, Fall 2016, McGill University

## Usage

The entry point of the program is [runner.py](src/runner.py), where all the
applications can be run by these simple flags :

```
usage: runner.py [-h] [-s SUBMISSION] [-lr] [-nv] [-sgd] [-rand] [-erf] [-bag]
                 [-vot] [-ada] [-svm]

optional arguments:
  -h, --help            show this help message and exit
  -lr, --logistic       logistic Regression
  -nv, --naive          Naive Bayes
  -sgd, --sgdclassifier
                        SGD Classifier
  -rand, --randomf      Random Forest
  -erf, --extrarf       Extra Random Forest
  -bag, --bagging       Bagging Ensemble
  -vot, --voting        Voting Ensemble
  -ada, --adaboost      Adaboost
  -svm, --svm           Linear SVM

  ```

K Nearest Neighbors implementation was done in R so use
[RStudio](https://www.rstudio.com/) to run it.

## Authors

- [Jeremy Georges-Filteau](https://github.com/jgeofil) - McGill ID : 260713547, _jeremy.georges-filteau@mail.mcgill.ca_
- [Yu Luo](https://github.com/yumcgill) - McGill ID : 260605878, _yu.t.luo@mail.mcgill.ca_
- [Koustuv Sinha](https://github.com/koustuvsinha) - McGill ID: 260721248, _koustuv.sinha@mail.mcgill.ca_
