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

num_features = 0

with open(TEST_FILE) as ftst:
    print 'Counting features'
    for line in ftst:
        features = line.split(',')
        num_feats = len(features)
        
ftst.close()

Feature_Names = []
FX = []
if os.path.exists(TEST_NP_ARRAY + '.npy'):
    print 'loading test numpy array.'
    XT = np.load(TEST_NP_ARRAY + '.npy')
    print 'test array loading finished.'
else:
    print 'generating test numpy array.'
    with open(TEST_FILE) as ftst:
        line_count = 0
        for line in ftst:
            features = line.split(',')
        
            F = []
            for i,f in enumerate(features):
                if line_count == 0:
                    Feature_Names.append(f.strip())
                else:
                    if '-' in f:
                        try:
                            dt = dparser.parse(f)
                            t = time.mktime(dt.timetuple())
                        except Exception,e:
                            print e
                            t = 0.0 
                    else:
                        if f != '':
                            t = float(f.strip())
                        else:
                            t = 0.0
                        F.append(t)
            FX.append(F)
            line_count +=1
            
#             if line_count == 5:
#                 break
        
    ftst.close()
    FX = [e for e in FX if e]
    
    XT = np.array(FX, dtype=np.object)
    print XT
    np.save(TEST_NP_ARRAY, XT)
    
    print 'test numpy generation and save done.'

FX = []
if os.path.exists(TRAIN_NP_ARRAY + '.npy'):
    print 'loading train numpy array'
    X = np.load(TRAIN_NP_ARRAY + '.npy')
    print 'train array loading finished'
else:
    print 'train numpy array generation started.'
    with open(TRAIN_FILE) as ftr:
        line_count = 0
        for line in ftr:
            features = line.split(',')
        
            F = []
            for i,f in enumerate(features):
                if line_count == 0:
                    pass
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
        
    ftr.close()
    FX = [e for e in FX if e]

    X = np.array(FX, dtype=np.object)
    np.save(TRAIN_NP_ARRAY, X)
    print 'train numpy array generation and saving done.'

print X.shape 

if os.path.exists(MODEL_FILE):
    print 'loading model file'
    mf = open(MODEL_FILE)
    clf = pickle.load(mf)
    mf.close()
    print 'model file loading done'
else:
    print 'SVM model genration started'
    print X[0]
    print X[1]
    print X[0][20]
    X_train = np.array(X[:,0:len(Feature_Names)-2],dtype=np.float64)
    print X_train
    y_train = (X[:,-1:len(Feature_Names)-1])
    y_train  = np.array(y_train.reshape((y_train.shape[0],)), dtype=np.float64)
    print y_train
    
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    mf = open(MODEL_FILE, 'w') 
    pickle.dump(clf, mf)
    mf.close()
    print 'SVM model generation finished'
    
# print 'prediction started'
# X_test = XT
# y_pred = clf.predict(X_test)
# print 'prediction done'
# 
# print 'evaluation started'