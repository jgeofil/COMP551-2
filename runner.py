## Main Runner Script

import numpy as np
import processing as pr
from sklearn.linear_model import LogisticRegression
import argparse

TRAIN_IN = 'train_in.csv'
TRAIN_OUT = 'train_out.csv'
TEST_IN = 'test_in.csv'

class Runner():
    def __init__(self,finalSubmission='submission'):
        self.finalSub = finalSubmission
        self.prcs = pr.Process(trainInFile=TRAIN_IN,trainOutFile=TRAIN_OUT,testFile=TEST_IN)
        self.logisticRegression()
        self.naiveBayes()
        self.knn()

    def logisticRegression(self):
        print 'Predicting logisticRegression'
        clf = LogisticRegression(C=1.25, n_jobs=-1,max_iter=1000,solver='lbfgs',class_weight='balanced')
        clf.fit(self.prcs.X_train,self.prcs.y_train)
        yp = clf.predict(self.prcs.X_test)
        acc = np.mean(yp == self.prcs.y_test)
        print 'Accuracy LogisticRegression %f' % acc
        print 'Predicting test file'
        y_fp = clf.predict(self.prcs.X_p)
        self.prcs.submit(y_fp,self.finalSub + '-LR.csv')

    def naiveBayes(self):
        ## call code by jgeof
        pass

    def knn(self):
        ## call code by terry
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--submission',type=str, help='Submission File',required=False)
    args = parser.parse_args()
    if args.submission:
        Runner(args.submission)
    else:
        Runner()
