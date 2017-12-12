import numpy as np
import matplotlib.pyplot as plt
import sys

x = np.linspace(5, 20, 20)[:, None]
y = 4.527355366 * np.log(x) - 2.418073037
# 8.850874152 ln(x) - 16.03609473
plt.plot(x, y,'-')
plt.show()
