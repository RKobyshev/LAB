import matplotlib.pyplot as plt
import numpy as np
import math

p0 = 265.4
p01 = 417.3
pgh = p01-p0
p = [417.3, 415.6, 413.7, 411.7, 407.8, 405.9, 402.0, 398.1, 392.2]
P = [x - pgh for x in p]
t = [21.1, 25.3, 30.2, 35.3, 40.2, 45.2, 50.0, 55.2, 60.2]
T = [(x + 273.15) for x in t]
R = 0.0005
sigma = [(x*R)/2 for x in P]
plt.scatter(T, sigma)
a, b = np.polyfit(T, sigma, 1)
x = np.linspace(min(T), max(T), 100)
y = a*x + b
plt.plot(x, y)
plt.show()