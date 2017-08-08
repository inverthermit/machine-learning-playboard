from workshop1 import *
import numpy as np
import matplotlib.pyplot as plt
import sys

# if sys.version_info[0] >= 3:
#     from urllib.request import urlretrieve
# else:
#     from urllib import urlretrieve
#
# #url = "https://staffwww.dcs.shef.ac.uk/people/T.Cohn/campus_only/mlai13/spambase.data.data"
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
# urlretrieve(url, 'spambase.data.data')


# load the CSV file as an array
Xandt = np.loadtxt('spambase.data.data', delimiter=',')
# randomly shuffle the rows, so as to remove any order bias
np.random.shuffle(Xandt)

# the last column are the response labels (targets), 0 = not spam, 1 = spam
# remap into -1 and +1 and take only the first 500 examples
t = Xandt[:500,-1] * 2 - 1
# and the remaining columns are the data
X = Xandt[:500,:-1]
print("Loaded", X.shape, "data points, and", t.shape, "labels,",  t.min() , t.max())
tmp = np.mean(X, 0)
#print(tmp)
print('Num of features : ' , len(tmp))

X = (X - np.mean(X, 0)) / np.std(X, 0)

# simple validation
# for k in [1,3,9,15,33,77]: #,3,9,15,33,77
#     Y = knn(X, X, t, k)
#     # print(np.where(t*Y<0),' LENGTH:',(np.where(t*Y<0)).length)
#     # print(np.where(t*Y>0),' LENGTH:',(np.where(t*Y>0)).length)
#     err = sum(i < 0 for i in t*Y)
#     correct = sum(i > 0 for i in t*Y)
#     errRate = err/(err+correct)
#     print(k,'-nn ' , errRate)
kArea = [1,3,9,15,33,77]
def evaluation(test_x, train_x, train_t, test_t):
    result = []
    for k in kArea: #,3,9,15,33,77
        Y = knn(test_x, train_x, train_t, k)
        # print(np.where(t*Y<0),' LENGTH:',(np.where(t*Y<0)).length)
        # print(np.where(t*Y>0),' LENGTH:',(np.where(t*Y>0)).length)
        err = sum(i < 0 for i in test_t*Y)
        correct = sum(i > 0 for i in test_t*Y)
        errRate = err/(err+correct)
        #print(k,'-nn ' , errRate)
        result.extend([errRate])
    return result

def heldout():
    N = X.shape[0]
    cut = int(N/2)
    Xtrain = X[:cut,:]
    ttrain = t[:cut]
    Xtest = X[cut:,:]
    ttest = t[cut:]
    print("There are", Xtrain.shape, "training samples, and", Xtest.shape, "heldout test samples")
    heldoutErr = evaluation(Xtest, Xtrain, ttrain, ttest)
    trainErr = evaluation(Xtrain, Xtrain, ttrain, ttrain)
    for i, k in enumerate(kArea):
        print(k,'-nn training error ',trainErr[i],' heldout error ',heldoutErr[i])

# Because if not use LOO, the nearest neighbours always consider the point itself.
# It should exclude itself from the nearest neighbous. This is the most important point.
def knn_LOO(x, t, k):
    predict = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        ns = neighbours(x, x[i], k+1).tolist()
        #print(ns)
        for j,val in enumerate(ns):
            if(ns[j] == i):
                ns.pop(j)
                #print(ns)
                break
        predict[i] = np.sign(np.sum(t[ns]))#sign: return the sign of a number
    return predict

# for k in [1,3,9,15,33,77]:
#     print('%d-nn' % k,
#           'LOO error', np.sum(knn_LOO(X, t, k) != t) / float(X.shape[0]))

pred = knn_LOO(X, t, 9)
print('true positives ', np.sum(t[pred == 1] == 1))
print('false positives', np.sum(t[pred == -1] == -1))# IMPLEMENT ME
print('true negatives ', np.sum(t[pred == 1] == -1))# IMPLEMENT ME
print('false negatives', np.sum(t[pred == -1] == 1))# IMPLEMENT ME

fprs = []
tprs = []

for k in [1,2,3,5,9,15,33,77]:
    pred = knn_LOO(X, t, k)

    tp = np.sum(t[pred==1] == 1)
    fp = np.sum(t[pred==1] == -1)
    tn = np.sum(t[pred==-1] == -1)
    fn = np.sum(t[pred==-1] == 1)

    # now compute the TPR and FPR
    fpr = fp / float(fp + tn)
    tpr = tp / float(tp + fn)


    fprs.append(fpr)
    tprs.append(tpr)

plt.plot(fprs, tprs, 'r-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
