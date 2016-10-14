import nltk as nl
import numpy as np
import pandas as pd
import sklearn as sk
import math

SMOOTHING = 1
MAX_FEATURES = 100

class NaiveB:
    def __init__(self):
        return

    def calculateProbabilities(self):
        self.categories = np.unique(self.y)
        # Discrete data
        self.thetasPerCat = []
        self.rowSums = self.X.sum(axis=1)
        for idx, val in enumerate(self.categories):
            #print('Doing category: ' + str(val))
            self.thetasPerCat.append(self.getThetasForCat(val))

    def getThetasForCat(self, k):
        featureProbabilities = []
        size = len(np.squeeze(self.X[0].A))
        count = 0
        for col in self.X.T:
            count += 1
            #if count%1000 == 0:
            #    print(str(count)+' of '+str(size)+' features..')
            col = np.squeeze(col.A)
            areNotCatSumForFeature = col[self.y!=k].sum() + SMOOTHING
            areNotCatSum = self.rowSums[self.y!=k].sum() + len(col) * SMOOTHING
            featureProbabilities.append(math.log(areNotCatSumForFeature/areNotCatSum))
        return featureProbabilities


    def train (self, X, y):
        tfid = sk.feature_extraction.text.TfidfVectorizer(stop_words='english', max_features = MAX_FEATURES)
        tfidfTrainSet = tfid.fit_transform(X[:,1])
        self.cv = sk.feature_extraction.text.CountVectorizer(stop_words='english', max_features = MAX_FEATURES).fit(X[:,1])
        #print('--- TRAINING ---')
        self.X = tfidfTrainSet
        self.y = y[:,1]
        self.calculateProbabilities()

    def predict (self, X):
        rawCountValidateSet = self.cv.transform(X[:,1])
        #print('--- PREDICTING ---')
        out = pd.DataFrame()
        i = 0
        for col in rawCountValidateSet:
            for idx, k in enumerate(self.categories):
                out.loc[i,k] = np.multiply(np.squeeze(col.A), self.thetasPerCat[idx]).sum()
            #if i % 1000 == 0:
            #    print(str(i)+' entries.. ')
            i += 1
        argmin = out.idxmin(axis=1)
        return argmin

    def calculateError(self, result, Y):
        return (Y[:,1] == result).sum()/float(len(Y))

    def crossValidation (self, X, y, k):
        perm = np.random.permutation(len(X))
        X, y = np.array_split(X[perm], k),np.array_split(y[perm], k)
        trainError = 0
        validateError = 0
        for i in range(k):
            mask = [True]*k
            mask[i] = False
            trainDataX, trainDataY = np.concatenate(np.array(X)[mask]), np.concatenate(np.array(y)[mask])
            validateDataX, validateDataY = X[i], y[i]
            self.train(trainDataX, trainDataY)
            validateResult = self.predict(validateDataX)
            trainResult = self.predict(trainDataX)
            trainE = self.calculateError(trainResult, trainDataY)
            validateE = self.calculateError(validateResult, validateDataY)
            trainError += trainE
            validateError += validateE

        print('--- Training ---')
        print('Error rate:')
        print(1 - trainError/float(k))
        print('--- Validation ---')
        print('Error rate:')
        print(1 - validateError/float(k))

    def confusionMatrix(self, trainX, trainY, valX, valY):
        self.train(trainX, trainY)
        arg = self.predict(valX)
        arg = np.array(arg)
        valY = np.array(valY[:,1])
        print(((arg == 'cs' ) & ( valY == 'cs')))
        print(((arg == 'cs' ) & ( valY == 'cs')).sum())
        print(((arg == 'math' )&( valY == 'cs')).sum())
        print(((arg == 'physics' )&( valY == 'cs')).sum())
        print(((arg == 'stat' )&( valY == 'cs')).sum())

        print(((arg == 'cs' )&( valY == 'math')).sum())
        print(((arg == 'math' )& (valY == 'math')).sum())
        print(((arg == 'physics' )&( valY == 'math')).sum())
        print(((arg == 'stat' )&( valY == 'math')).sum())

        print(((arg == 'cs' )&( valY == 'physics')).sum())
        print(((arg == 'math' )&( valY == 'physics')).sum())
        print(((arg == 'physics' )&( valY == 'physics')).sum())
        print(((arg == 'stat' )&( valY == 'physics')).sum())

        print(((arg == 'cs' )&( valY == 'stat')).sum())
        print(((arg == 'math' )&( valY == 'stat')).sum())
        print(((arg == 'physics' )&( valY == 'stat')).sum())
        print(((arg == 'stat' )&( valY == 'stat')).sum())


#inSet = np.array(pd.read_csv('train_in.csv', quotechar='"', skipinitialspace=True, header=0))
#outSet = np.array(pd.read_csv('train_out.csv', quotechar='"', skipinitialspace=True, header=0))
#test = np.array(pd.read_csv('test_in.csv', quotechar='"', skipinitialspace=True, header=0))

#size = len(inSet)
#cuttof = 80000
#vallen = 10000
#trainSet = inSet[:cuttof,:]
#validateSet = inSet[cuttof+1:cuttof+vallen,:]
#categoriesTrain = outSet[:cuttof,:]
#categoriesValidate = outSet[cuttof+1:cuttof+vallen,:]


#nb = NaiveB()
#nb.crossValidation(trainSet, categoriesTrain, 5)

#nb.confusionMatrix(trainSet, categoriesTrain, validateSet, categoriesValidate)


#arg.to_csv('out.csv')
#print((categoriesValidate[:,1] == arg).sum()/float(vallen))
