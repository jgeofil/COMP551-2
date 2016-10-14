## Main Runner Script
## KNN implementation is made in R and has been provided in knearest.R
## for reference

import numpy as np
import processing as pr
import pandas as pd
import NB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from time import time
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.svm import LinearSVC
import argparse

TRAIN_IN = 'data/train_in.csv'
TRAIN_OUT = 'data/train_out.csv'
TEST_IN = 'data/test_in.csv'

class Runner():
    def __init__(self,args=None):
        self.finalSub = 'submission'
        self.prcs = pr.Process(trainInFile=TRAIN_IN,trainOutFile=TRAIN_OUT,testFile=TEST_IN)
        if args.naive:
            self.naiveBayes()
        estimatorDict = {}
        if args.logistic:
            estimatorDict['LogisticRegression'] = LogisticRegression(C=1.25, n_jobs=-1,max_iter=1000,solver='lbfgs',class_weight='balanced')
        if args.sgdclassifier:
            estimatorDict['SGDClassifier'] = SGDClassifier(loss='modified_huber',penalty='l2',n_jobs=-1,learning_rate='optimal',n_iter=500,alpha=0.00004)
        if args.randomf:
            estimatorDict['RandomForest'] = RandomForestClassifier(n_jobs=-1,n_estimators=100, class_weight='balanced', max_features='log2')
        if args.extrarf:
            estimatorDict['ExtraRandomTree'] = ExtraTreesClassifier(n_estimators=100,n_jobs=-1)
        if args.bagging:
            estimatorDict['Bagging'] = BaggingClassifier(base_estimator=estimatorDict['ExtraRandomTree'])
        if args.voting:
            estimatorDict['Voting'] = VotingClassifier(estimators=[('lr',estimatorDict['LogisticRegression']),('naive',NB.NaiveB()),('sgdc',estimatorDict['SGDClassifier'])])
        if args.adaboost:
            estimatorDict['AdaBoost'] = AdaBoostClassifier(base_estimator=estimatorDict['RandomForest'],n_estimators=100,learning_rate=0.001)
        if args.svm:
            self.selectChiBest(1000)
            estimatorDict['LinearSVM'] = LinearSVC()

        self.runClassifications()

    def naiveBayes(self):
        print 'Predicting naiveBayes'
        trainX = np.array(pd.read_csv(TRAIN_IN, quotechar='"', header=0))
        trainY = np.array(pd.read_csv(TRAIN_OUT, quotechar='"', header=0))
        test = np.array(pd.read_csv(TEST_IN, quotechar='"', header=0))
        nb = NB.NaiveB()
        nb.train(trainX, trainY)
        print 'Predicting test file'
        arg = nb.predict(test)
        out = pd.DataFrame(columns=['id', 'category'])
        out['id'], out['category'] = arg.index, arg
        out.to_csv(self.finalSub + '-NB.csv',index=False)


    def runClassifications(self,estimatorDict):
        for estimatorName,estimator in estimatorDict:
            print estimatorName
            clf = estimator
            if args.svm:
                clf.fit(self.chi.X_train,self.prcs.y_train)
                yp = clf.predict(self.chi.X_test)
            else:
                clf.fit(self.prcs.X_train,self.prcs.y_train)
                yp = clf.predict(self.prcs.X_test)
            acc = np.mean(yp == self.prcs.y_test)
            print 'Accuracy %f' % acc
            print 'Predicting test file'
            if args.svm:
                y_fp = clf.predict(self.chi.X_p)
            else:
                y_fp = clf.predict(self.prcs.X_p)
            self.prcs.submit(y_fp,self.finalSub + '-' + estimatorName + '.csv')

    def selectChiBest(self,k):
        print("Extracting %d best features by a chi-squared test" %k)
        t0 = time()
        ch2 = SelectKBest(chi2, k=sel)
        self.chi.X_train = ch2.fit_transform(self.prcs.X_train, self.prcs.y_train)
        self.chi.X_test = ch2.transform(self.prcs.X_test)
        self.chi.X_p = ch2.transform(self.prcs.X_p)
        # keep selected feature names
        self.chi.feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--submission',type=str, help='Submission File',required=False)
    parser.add_argument('-lr','--logistic',help='logistic Regression',action='store_true')
    parser.add_argument('-nv','--naive',help='Naive Bayes',action='store_true')
    parser.add_argument('-sgd','--sgdclassifier',help='SGD Classifier',action='store_true')
    parser.add_argument('-rand','--randomf',help='Random Forest',action='store_true')
    parser.add_argument('-erf','--extrarf',help='Extra Random Forest',action='store_true')
    parser.add_argument('-bag','--bagging',help='Bagging Ensemble',action='store_true')
    parser.add_argument('-vot','--voting',help='Voting Ensemble',action='store_true')
    parser.add_argument('-ada','--adaboost',help='Adaboost',action='store_true')
    parser.add_argument('-svm','--svm',help='Linear SVM',action='store_true')
    args = parser.parse_args()
    # Run the classifications
    Runner(args=args)
