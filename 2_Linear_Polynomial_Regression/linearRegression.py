import numpy as np
import matplotlib.pyplot as plt
import sys
csv = """1896,4.47083333333333
1900,4.46472925981123
1904,5.22208333333333
1908,4.1546786744085
1912,3.90331674958541
1920,3.5695126705653
1924,3.8245447722874
1928,3.62483706600308
1932,3.59284275388079
1936,3.53880791562981
1948,3.6701030927835
1952,3.39029110874116
1956,3.43642611683849
1960,3.2058300746534
1964,3.13275664573212
1968,3.32819844373346
1972,3.13583757949204
1976,3.07895880238575
1980,3.10581822490816
1984,3.06552909112454
1988,3.09357348817
1992,3.16111703598373
1996,3.14255243512264
2000,3.08527866650867
2004,3.1026582928467
2008,2.99877552632618
2012,3.03392977050993"""


if sys.version_info[0] >= 3:
    import io # Python3
    olympics = np.genfromtxt(io.BytesIO(csv.encode()), delimiter=",")
else:
    from StringIO import StringIO  # Python2
    olympics = np.genfromtxt(StringIO(csv), delimiter=',') #Python 2

#print(olympics)
x = olympics[:, 0:1] # two dimentional array, first : is to get all olympics[x], second is get olympics[x][0]
y = olympics[:, 1:2]
# print(x)
# print(y)
# plt.plot(x,y, 'rx')
#plt.show()
b = -0.4
# print(len(x))
a = sum(y-b*x)/len(x)
b = sum((y-a)*x)/sum(np.square(x))
x_test = np.linspace(1890, 2020, 130)[:, None]
# print(x_test)
f_test = b*x_test + a
# plt.plot(x_test, f_test, 'b-')
# plt.plot(x, y, 'rx')
#plt.show()
SSR =  sum(np.square(y-a-b*x)) # over to you
# print(SSR)

def iterativeSolution():
    for i in np.arange(10000):
        a = sum(y-b*x)/len(x) # np.mean(y-b*x)
        b = sum((y-a)*x)/sum(np.square(x)) # ((y-a)*x).sum()/(x**2).sum()
        SSR = sum(np.square(y-a-b*x))
        if i % 500 == 0:
            print('Iteration# ' ,i ,', training error SSR',SSR)
    (a, b)
    f_test = b*x_test + a
    plt.plot(x_test, f_test, 'b-')
    plt.plot(x, y, 'rx')
    plt.show()

X = np.hstack((np.ones_like(x), x))
# print(X)

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)) # back to you
print(w)

a, b = w
f_test = b*x_test + a
plt.plot(x_test, f_test, 'b-')
plt.plot(x, y, 'rx')

SSR = sum(np.square(y-a-b*x)) # back to you
















#
