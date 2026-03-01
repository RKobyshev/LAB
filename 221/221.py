import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

data45 = pd.read_csv('20260219_1771513766824_45.csv')
data60 = pd.read_csv('20260219_1771515236479_60.csv')
data80 = pd.read_csv('20260219_1771516314459_81.5.csv')
data100 = pd.read_csv('20260219_1771517279990_100.2.csv')
data120 = pd.read_csv('20260219_1771518254324_118.8.csv')
data160 = pd.read_csv('20260219_1771519500497_159.6.csv')

t45 = data45.iloc[:, 0].tolist()
t60 = data60.iloc[:, 0].tolist()
t80 = data80.iloc[:, 0].tolist()
t100 = data100.iloc[:, 0].tolist()
t120 = data120.iloc[:, 0].tolist()
t160 = data160.iloc[:, 0].tolist()

u45 = data45.iloc[:, 1].tolist()
u60 = data60.iloc[:, 1].tolist()
u80 = data80.iloc[:, 1].tolist()
u100 = data100.iloc[:, 1].tolist()
u120 = data120.iloc[:, 1].tolist()
u160 = data160.iloc[:, 1].tolist()

U45l = [math.log(x/1000) for x in u45]
U60l = [math.log(x/1000) for x in u60]
U80l = [math.log(x/1000) for x in u80]
U100l = [math.log(x/1000) for x in u100]
U120l = [math.log(x/1000) for x in u120]
U160l = [math.log(x/1000) for x in u160]

plt.plot(t45, U45l)
plt.plot(t60, U60l)
plt.plot(t80, U80l)
plt.plot(t100, U100l)
plt.plot(t120, U120l)
plt.plot(t160, U160l)
plt.title("Зависимость напряжения от времени в полулогарифмическом масштабе")
plt.xlabel("Время, с")
plt.ylabel("Логарифм напряжения, В")
plt.grid(alpha=0.5)
plt.show()

T45 = (-1)/(np.polyfit(t45, U45l, 1)[0])
print(T45)
T60 = (-1)/(np.polyfit(t60, U60l, 1)[0])
print(T60)
T80 = (-1)/(np.polyfit(t80, U80l, 1)[0])
print(T80)
T100 = (-1)/(np.polyfit(t100, U100l, 1)[0])
print(T100)
T120 = (-1)/(np.polyfit(t120, U120l, 1)[0])
print(T120)
T160 = (-1)/(np.polyfit(t160, U160l, 1)[0])
print(T160)

v = 1200
D = []
ls = 5.5
d45 = v*ls/(2*T45)
D.append(d45)
d60 = v*ls/(2*T60)
D.append(d60)
d80 = v*ls/(2*T80)
D.append(d80)
d100 = v*ls/(2*T100)
D.append(d100)
d120 = v*ls/(2*T120)
D.append(d120)
d160 = v*ls/(2*T160)
D.append(d160)
D1 = [d60, d80, d100, d120, d160]
P = [1/45, 1/60, 1/80, 1/100, 1/120, 1/160]
P1 = [1/60, 1/80, 1/100, 1/120, 1/160]
plt.scatter(P, D)
c = np.polyfit(P, D, 1)
a = c[0]
b = c[1]
print(a, b)
D760 = a*(1/760)
print(D760)
c1 = np.polyfit(P1, D1, 1)
print(c1[0]/760)
x = np.linspace(min(P), max(P), 100)
y = a*x + b
plt.plot(x, y)
plt.grid(alpha=0.3)
plt.title("Зависимость коэффициента диффузии от обратного давления")
plt.xlabel("1/давление, 1/торр")
plt.ylabel("Коэффициент диффузии, см /с")
plt.show()

v = ((8*8.314*(273+23))/(3.1415*0.004))**0.5
L = [(3*x)/v for x in D]
PATM = [760*x for x in P]
n = [x/((1.3806488)*(10**(-23))*(273+23)) for x in PATM]
s = []
for i in range(len(n)):s.append(n[i]/L[i])
print(L, "\n", n, "\n", s)