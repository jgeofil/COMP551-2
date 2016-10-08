import nltk as nl
import numpy as np
import pandas as pd
import sklearn as sk
import math


train = np.array(pd.read_csv('train_in.csv', quotechar='"', skipinitialspace=True))[:4000,:]
cats = np.array(pd.read_csv('train_out.csv', quotechar='"', skipinitialspace=True))[:4000,:]
test = np.array(pd.read_csv('train_in.csv', quotechar='"', skipinitialspace=True))[5001:6000,:]
testC = np.array(pd.read_csv('train_out.csv', quotechar='"', skipinitialspace=True))[5001:6000,:]
stopwords = nl.corpus.stopwords.words('english')
cats = cats[:,1]
testC = testC[:,1]
tfid = sk.feature_extraction.text.TfidfVectorizer(stop_words='english')
tfidf_matrix = tfid.fit_transform(train[:,1]).todense()

rawcount = sk.feature_extraction.text.CountVectorizer(stop_words='english')
rawcount.fit(train[:,1])
test_matrix =  rawcount.transform(test[:,1]).todense()

#test_matrix =  tfid.transform(test[:,1]).todense()

class NaiveB:
    def __init__(self):
        return

    def calculateProbabilities(self):
        self.possibleOutcomes = np.unique(self.y)
        # Discrete data
        self.kProbabilities = pd.Series()
        self.colSums = self.X.sum(axis=1)
        for k in self.possibleOutcomes:
            self.kProbabilities.loc[k] = self.getProbabilitiesForK(k)

    def getProbabilitiesForK(self, k):
        featureProbabilities = []
        for m in range(0,self.m): # For every discrete feature
            l = self.X.T.A[m]
            inCat = l[self.y!=k].sum() + 1
            denom = self.colSums[self.y!=k].sum() + self.m
            featureProbabilities.append(math.log(inCat/denom))
        return featureProbabilities


    def train (self, X, y):
        # Number of features in model
        self.m = X[0].size
        print(self.m)
        # Size of dataset
        self.n = X.size
        self.X = X
        self.y = y

        self.calculateProbabilities()

    def predict (self, X):

        out = pd.DataFrame()
        for x in range(0, len(X.A)):
            for k in self.possibleOutcomes:
                out.loc[x,k] = np.multiply(X.A[x], self.kProbabilities[k]).sum()
        argmin = out.idxmin(axis=1)
        return argmin


nb = NaiveB()
nb.train(tfidf_matrix, cats)
arg = nb.predict(test_matrix)
print((testC == arg).sum())
