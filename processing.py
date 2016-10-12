## Processing scripts for the data

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

class Process:
    def __init__(self,trainInFile='train_in.csv',trainOutFile='train_out.csv',testFile='test_in.csv'):
        self.df = pd.read_csv(trainInFile)
        self.df_y = pd.read_csv(trainOutFile)
        self.df_p = pd.read_csv(testFile)
        self.clean()
        self.labelEncode()
        self.vectorize()

    def clean(self):
        dropVals = list(self.df_y[self.df_y.category == 'category']['id'])
        self.df = self.df[~self.df.id.isin(dropVals)]
        self.df_y = self.df_y[~self.df_y.id.isin(dropVals)]

    def labelEncode(self):
        print 'Encoding labels'
        self.le = preprocessing.LabelEncoder()
        self.le.fit(['cs','math','physics','stat'])
        self.y = self.le.transform(self.df_y.category)

    def submit(self,y,filename='submission.csv'):
        print 'Decoding labels'
        y_act = self.le.inverse_transform(y)
        self.df_p['category'] = y_act
        self.df_p.drop(['abstract'],axis=1,inplace=True)
        print 'Predicted rows = %d' % len(self.df_p)
        self.df_p.to_csv(filename,index=False)

    def sent_to_wordlist(self, sent, remove_stopwords=False ):
        sent = re.sub("[^a-zA-Z]"," ", sent)
        words = sent.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return(words)

    def vectorize(self):
        print 'Starting text processing'
        wordnet_lemmatizer = WordNetLemmatizer()
        X_train, X_test, y_train, y_test = train_test_split(self.df.abstract,self.y)
        X_train_c = []
        X_test_c = []
        for s in X_train:
            X_train_c.append( " ".join( self.sent_to_wordlist( s )))
        for s in X_test:
            X_test_c.append( " ".join( self.sent_to_wordlist( s )))
        X_p = []
        for s in self.df_p['abstract']:
            X_p.append( " ".join(self.sent_to_wordlist(s)))
        stops = set(stopwords.words("english"))
        X_train_w = []
        for sent in X_train_c:
            words = sent.lower().split()
            words = [w for w in words if not w in stops]
            # lemmatization
            words = [wordnet_lemmatizer.lemmatize(w) for w in words]
            X_train_w.append(words)
        X_test_w = []
        for sent in X_test_c:
            words = sent.lower().split()
            words = [w for w in words if not w in stops]
            # lemmatization
            words = [wordnet_lemmatizer.lemmatize(w) for w in words]
            X_test_w.append(words)
        X_p_w = []
        for sent in X_p:
            words = sent.lower().split()
            words = [w for w in words if not w in stops]
            # lemmatization
            words = [wordnet_lemmatizer.lemmatize(w) for w in words]
            X_p_w.append(words)
        X_train_ws = []
        for sent in X_train_w:
            X_train_ws.append(" ".join(sent))
        X_test_ws = []
        for sent in X_test_w:
            X_test_ws.append(" ".join(sent))
        X_p_ws = []
        for sent in X_p_w:
            X_p_ws.append(" ".join(sent))
        print 'Text processing complete'
        tfidf_vectorizer = TfidfVectorizer( max_features = 40000, ngram_range = ( 1, 3 ), sublinear_tf = True )
        print 'Vectorizing Tfidf ...'
        X_train_tf = tfidf_vectorizer.fit_transform(X_train_ws)
        X_test_tf = tfidf_vectorizer.transform(X_test_ws)
        X_p_tf = tfidf_vectorizer.transform(X_p_ws)
        print 'Vectorizing complete'
        self.X_train = preprocessing.normalize(X_train_tf, norm='l2')
        self.X_test = preprocessing.normalize(X_test_tf, norm='l2')
        self.X_p = preprocessing.normalize(X_p_tf, norm='l2')
        self.y_train = y_train
        self.y_test = y_test
