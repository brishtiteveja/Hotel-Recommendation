import os
import numpy as np
import matplotlib as mlp
import sklearn as sk
import time 
import datetime
import dateutil.parser as dparser
import pickle
from sklearn import svm


DATA_DIR = '../data/'
TRAIN_FILE = DATA_DIR + 'train.csv'
TEST_FILE = DATA_DIR + 'test.csv'

TRAIN_NP_ARRAY= DATA_DIR + 'train.np'
TEST_NP_ARRAY = DATA_DIR + 'test.np'

MODEL_FILE = DATA_DIR + 'linear_svm_clf.model'

FeaturesNames = []
FX = []
with open(TRAIN_FILE) as ftr:
    line_count = 0
    for line in ftr:
        features = line.split(',')
        
        F = []
        for i,f in enumerate(features):
            if line_count == 0:
                FeaturesNames.append(f.strip())
            else:
                if '-' in f:
                    try:
                        dt = dparser.parse(f)
                        t = time.mktime(dt.timetuple())
                    except Exception,e:
                        print e  # day is out of range error needs to be fixed
                        t = 0.0 
                else:
                    if f != '':
                        t = float(f.strip())
                    else:
                        t = 0.0
                    F.append(t)
        FX.append(F)
        line_count +=1
        if line_count == 100:
            break
        
    ftr.close()
    FX = [e for e in FX if e]

    X = np.array(FX, dtype=np.object)
    print 'train numpy array generation and saving done.'

print X.shape 
print X[1]

if os.path.exists(MODEL_FILE):
    print 'loading model file'
    mf = open(MODEL_FILE)
    clf = pickle.load(mf)
    mf.close()
    print 'model file loading done'
else:
    print 'SVM model genration started'
    print X.shape
    X_train = np.array(X[:,0:X.shape[1]-1],dtype=np.float64)
    y_train = (X[:,-1:X.shape[1]])
    y_train  = np.array(y_train.reshape((y_train.shape[0],)), dtype=np.float64)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    mf.close()
    print 'SVM model generation finished'
    
# print 'prediction started'
# X_test = XT
# y_pred = clf.predict(X_test)
# print 'prediction done'
# 
# print 'evaluation started'