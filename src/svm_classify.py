from jpype import *
from sklearn import svm, preprocessing, metrics
import numpy as np
from sklearn.cross_validation import KFold
from scipy.sparse import csr_matrix
import pickle
import optparse


def svm_train(train_file, num_feats, model_file, clf_kernel='linear',):
    X_train = []
    y_train = []
    
    print "Parsing train file", train_file
    with open(train_file) as fp:
        for line in fp:
            data = line.split()
            y = int(float(data[0]))
            y_train.append(y)
            feat_vector = [0.0]*(num_feats)
            for i in xrange(1, len(data)):
                if "#" in data[i]:
                    break
                (index, value) = data[i].split(":")
                feat_vector[int(index) - 1] = float(value)
            
            X_train.append(feat_vector)

    print "Parsed train examples:", len(X_train)
    print "Number of classes:", len(set(y_train))
    print "Number of features:", len(X_train[0])

    print "Replacing NaN with 0 and inf with finite numbers..."
    X_train = csr_matrix(np.nan_to_num(np.asarray(X_train)))
    y_train = np.asarray(y_train)
    
    clf = svm.SVC(kernel=clf_kernel)
    clf.fit(X_train, y_train)
    
    mf = open(model_file, 'w') 
    pickle.dump(clf, mf)
    mf.close()

def svm_testDev(test_file, num_feats, model_file):
    X_test = []
    y_test = []
    print "Parsing test/dev file", test_file
    with open(test_file) as fp:
        for line in fp:
            data = line.split()
            y = int(float(data[0]))
            y_test.append(y)
            feat_vector = [0.0]*(num_feats)
            for i in xrange(1, len(data)):
                if "#" in data[i]:
                    break
                (index, value) = data[i].split(":")
                feat_vector[int(index) - 1] = float(value)
            X_test.append(feat_vector)

    print "Parsed test/Dev examples:", len(X_test)
    print "Number of classes:", len(set(y_test))
    print "Number of features:", len(X_test[0])

    print "Replacing NaN with 0 and inf with finite numbers..."
    X_test = csr_matrix(np.nan_to_num(np.asarray(X_test)))

    y_test = np.asarray(y_test)
#     print "Feature matrix shape", X_test.shape

    
    mf = open(model_file)
    clf = pickle.load(mf)
    mf.close()

    y_pred = clf.predict(X_test)
    
    return y_test, y_pred

def svm_reportMetric(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print metrics.classification_report(y_test, y_pred)
    

def svm_test(test_file, num_feats, model_file):
    X_test = []
    ids = []
    
    print "Parsing test file", test_file
    with open(test_file) as fp:
        for line in fp:
            data = line.split()
            feat_vector = [0.0]*(num_feats)
            for i in xrange(1, len(data)):
                if "#" in data[i]:
                    if (i+ 4) == len(data) - 1:
                        id = data[i + 4] # getting relation id
                        ids.append(id)
                    break
                (index, value) = data[i].split(":")
                feat_vector[int(index) - 1] = float(value)
            X_test.append(feat_vector)
            
    
    print "Parsed train examples:", len(X_test)
    print "Number of features:", len(X_test[0])
    
    print "Replacing NaN with 0 and inf with finite numbers..."
    X_test = csr_matrix(np.nan_to_num(np.asarray(X_test)))

    mf = open(model_file)
    clf = pickle.load(mf)
    y_pred = clf.predict(X_test)
    mf.close()
    
    ids = map(int, ids)
    return ids, y_pred
    
def main():
    parser = optparse.OptionParser()
    parser.add_option("-i", dest='train_file', type='string')
    parser.add_option("-r", dest='test_file', type='string')
    parser.add_option("-f", dest='num_feats', type='int')
    parser.add_option("-m", dest='model_folder', type='string')
    (opts, args) = parser.parse_args()
    #svm_train(opts.train_file, opts.num_feats, opts.model_folder)
    #y_test, y_pred = svm_testDev(opts.test_file, opts.num_feats, opts.model_folder)
    #svm_reportMetric(y_test, y_pred)
    ids, y_pred = svm_test(opts.test_file, opts.num_feats, opts.model_folder)
    print ids

if __name__ == "__main__":
    main()
    
