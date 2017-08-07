#%matplotlib
import numpy as np
import matplotlib.pyplot as plt
# x = 0.4 * np.random.randn(40, 2) # gaussian points 40 x 2
# x[0:10] += np.array([1, 0])
# x[10:20] += np.array([0, 1])
# x[20:30] += np.array([0, 0])
# x[30:40] += np.array([1, 1])
# t = np.hstack([-np.ones(20), np.ones(20)]) # target vector 1 x 40
# print(x)
# print(t==-1,0)
# print(x[t==-1,0])
# print(t==-1,1)
# print(x[t==-1,1])

# plt.plot(x[t==-1,0], x[t==-1,1], 'b.')# x[t==-1,0] : if t[i]==-1 then x[i][0]
# plt.plot(x[t==1,0], x[t==1,1], 'r.')
#print(x.shape, np.mean(x), np.var(x), np.min(x), np.max(x))
#print(t.shape, np.mean(t), np.var(t), np.min(t), np.max(t))
#plt.show()



def euclidean(x, z):
    d = x - z
    if len(d.shape) > 1 and d.shape[1] > 1:
        return np.sqrt(np.diag(np.dot(d, d.T)))
    else:
        return np.sqrt(np.dot(d, d))

def neighbours(x, train_x, k):
    # IMPLEMENT ME to return the indices of the k closest elements to x in train_x
    dist = euclidean(x, train_x)
    return np.argsort(dist)[:k]


def knn(test_x, train_x, train_t, k):
    predict = np.zeros(test_x.shape[0])
    for i in range(test_x.shape[0]):
        ns = neighbours(train_x, test_x[i], k)
        #print(ns)
        predict[i] = np.sign(np.sum(train_t[ns]))#sign: return the sign of a number
    return predict

# tmp = euclidean(np.array([[0,0],[1,0], [0.5,0.5]]), np.array([1,1]))
# knn(x, x, t, 3)
#
# X1, X2 = np.meshgrid(np.arange(-0.7, 1.7, 0.025), np.arange(-0.7, 1.7, 0.025))
# X12 = np.column_stack([X1.flatten(), X2.flatten()])
# Y = knn(X12, x, t, 1)
# cs = plt.contour(X1, X2, Y.reshape(X1.shape), levels=[0])
# plt.plot(x[t==-1,0], x[t==-1,1], 'o')
# plt.plot(x[t==1,0], x[t==1,1], 'o')
#plt.show()
