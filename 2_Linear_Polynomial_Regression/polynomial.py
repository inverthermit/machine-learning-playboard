import numpy as np
import matplotlib.pyplot as plt
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
x = olympics[:, 0:1]
y = olympics[:, 1:2]
plt.plot(x, y, 'rx')
plt.show()

num_data = x.shape[0]
num_pred_data = 100 # how many points to use for plotting predictions
x_pred = linspace(1890, 2016, num_pred_data)[:, None] # input locations for predictions
order = 4 # The polynomial order to use.
print ('Num of training samples: ',num_data)
print('Num of testing samples: ',num_pred_data)

Phi = np.zeros((num_data, order+1))
Phi_pred = np.zeros((num_pred_data, order+1))
for i in range(0, order+1):
    Phi[:, i:i+1] = x**i
    Phi_pred[:, i:i+1] = x_pred**i

w = np.linalg.solve(np.dot(Phi.T, Phi), np.dot(Phi.T, y)) # back to you

f = Phi*w
f_pred = Phi_pred*w

SSR = sum((y-f)**2)
print(SSR)















#
