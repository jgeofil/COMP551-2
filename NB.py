import nltk as nl
import numpy as np
import pandas as pd
import sklearn as sk
import math

SMOOTHING = 1

class NaiveB:
    def __init__(self):
        return

    def calculateProbabilities(self):
        self.categories = np.unique(self.y)
        # Discrete data
        self.thetasPerCat = []
        self.rowSums = self.X.sum(axis=1)
        for idx, val in enumerate(self.categories):
            print('Doing category: ' + str(val))
            self.thetasPerCat.append(self.getThetasForCat(val))

    def getThetasForCat(self, k):
        featureProbabilities = []
        size = len(np.squeeze(self.X[0].A))
        count = 0
        for col in self.X.T:
            count += 1
            if count%1000 == 0:
                print(str(count)+' of '+str(size)+' features..')
            col = np.squeeze(col.A)
            areNotCatSumForFeature = col[self.y!=k].sum() + SMOOTHING
            areNotCatSum = self.rowSums[self.y!=k].sum() + len(col) * SMOOTHING
            featureProbabilities.append(math.log(areNotCatSumForFeature/areNotCatSum))
        return featureProbabilities


    def train (self, X, y):
        print('--- TRAINING ---')
        self.X = X
        self.y = y
        self.calculateProbabilities()

    def predict (self, X):
        print('--- PREDICTING ---')
        out = pd.DataFrame()
        i = 0
        for col in X:
            for idx, k in enumerate(self.categories):
                out.loc[i,k] = np.multiply(np.squeeze(col.A), self.thetasPerCat[idx]).sum()
            if i % 1000 == 0:
                print(str(i)+' entries.. ')
            i += 1
        argmin = out.idxmin(axis=1)
        return argmin


inSet = np.array(pd.read_csv('train_in.csv', quotechar='"', skipinitialspace=True, header=0))
outSet = np.array(pd.read_csv('train_out.csv', quotechar='"', skipinitialspace=True, header=0))
test = np.array(pd.read_csv('test_in.csv', quotechar='"', skipinitialspace=True, header=0))

size = len(inSet)
cuttof = 20000
vallen = 50
trainSet = inSet[:cuttof,:]
validateSet = inSet[cuttof+1:cuttof+vallen,:]
categoriesTrain = outSet[:cuttof,:]
categoriesValidate = outSet[cuttof+1:cuttof+vallen,:]

tfid = sk.feature_extraction.text.TfidfVectorizer(stop_words='english')
tfidfTrainSet = tfid.fit_transform(trainSet[:,1])#.todense()

cv = sk.feature_extraction.text.CountVectorizer(stop_words='english').fit(trainSet[:,1])
#rawCountValidateSet = cv.transform(validateSet[:,1]).todense()
rawCountValidateSet = cv.transform(test[:,1])#.todense()

nb = NaiveB()
nb.train(tfidfTrainSet, categoriesTrain[:,1])
arg = nb.predict(rawCountValidateSet)
print(arg)
arg.to_csv('out.csv')
#print((categoriesValidate[:,1] == arg).sum()/float(vallen))
