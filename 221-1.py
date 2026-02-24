import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Чтение данных из CSV-файлов
data45 = pd.read_csv('20260219_1771513766824_45.csv')
data60 = pd.read_csv('20260219_1771515236479_60.csv')
data80 = pd.read_csv('20260219_1771516314459_81.5.csv')
data100 = pd.read_csv('20260219_1771517279990_100.2.csv')
data120 = pd.read_csv('20260219_1771518254324_118.8.csv')
data160 = pd.read_csv('20260219_1771519500497_159.6.csv')

# Извлечение времени и напряжения
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

# Логарифмирование напряжения (деление на 1000 для перевода в вольты)
U45l = [math.log(x/1000) for x in u45]
U60l = [math.log(x/1000) for x in u60]
U80l = [math.log(x/1000) for x in u80]
U100l = [math.log(x/1000) for x in u100]
U120l = [math.log(x/1000) for x in u120]
U160l = [math.log(x/1000) for x in u160]

# Построение графиков логарифма напряжения от времени
plt.plot(t45, U45l, label='45 торр')
plt.plot(t60, U60l, label='60 торр')
plt.plot(t80, U80l, label='80 торр')
plt.plot(t100, U100l, label='100 торр')
plt.plot(t120, U120l, label='120 торр')
plt.plot(t160, U160l, label='160 торр')
plt.title("Зависимость напряжения от времени в полулогарифмическом масштабе")
plt.xlabel("Время, с")
plt.ylabel("Логарифм напряжения, В")
plt.grid(alpha=0.5)
plt.legend()
plt.show()

# Функция для получения τ и его погрешности
def get_tau_and_error(t, y):
    coeff, cov = np.polyfit(t, y, 1, cov=True)
    k, b = coeff
    dk = np.sqrt(cov[0, 0])
    tau = -1/k
    dtau = tau * dk/abs(k)   # относительная погрешность переносится
    return tau, dtau, k, dk

# Определение времени релаксации τ и погрешностей
tau45, dt45, k45, dk45 = get_tau_and_error(t45, U45l)
tau60, dt60, k60, dk60 = get_tau_and_error(t60, U60l)
tau80, dt80, k80, dk80 = get_tau_and_error(t80, U80l)
tau100, dt100, k100, dk100 = get_tau_and_error(t100, U100l)
tau120, dt120, k120, dk120 = get_tau_and_error(t120, U120l)
tau160, dt160, k160, dk160 = get_tau_and_error(t160, U160l)

print("τ, с:", tau45, tau60, tau80, tau100, tau120, tau160)
print("Погрешности τ, с:", dt45, dt60, dt80, dt100, dt120, dt160)

# Расчёт коэффициента диффузии D
V = 1200              # объём сосуда, см³
LS = 5.5              # отношение L/S, см⁻¹

def calc_D(tau):
    return V * LS / (2 * tau)

D45 = calc_D(tau45)
D60 = calc_D(tau60)
D80 = calc_D(tau80)
D100 = calc_D(tau100)
D120 = calc_D(tau120)
D160 = calc_D(tau160)

D = [D45, D60, D80, D100, D120, D160]
print("D, см²/с:", D)

# Погрешность D (через погрешность τ, пренебрегая геометрией)
dD = [d * (dt/tau) for d, dt, tau in zip(D, [dt45, dt60, dt80, dt100, dt120, dt160],
                                           [tau45, tau60, tau80, tau100, tau120, tau160])]
print("Погрешности D, см²/с:", dD)

# Давления (фактические значения, округлённые согласно именам файлов)
pressures = [45, 60, 80, 100, 120, 160]  # торр
P_inv = [1/p for p in pressures]

# --- Второй график с погрешностями ---
plt.errorbar(P_inv, D, yerr=dD, fmt='o', capsize=5, label='Экспериментальные точки')

# Линейная аппроксимация (МНК) для всех точек
coeff_all, cov_all = np.polyfit(P_inv, D, 1, cov=True)
a_all, b_all = coeff_all
da_all = np.sqrt(cov_all[0, 0])
print(f"a_all = {a_all:.3f} ± {da_all:.3f}, b_all = {b_all:.3f}")

# Экстраполяция к атмосферному давлению (760 торр) без свободного члена
D760_all = a_all / 760
print("D при 760 торр (все точки):", D760_all)

# Для набора без первой точки
P_inv1 = P_inv[1:]
D1 = D[1:]
coeff1, cov1 = np.polyfit(P_inv1, D1, 1, cov=True)
a1, b1 = coeff1
print(f"a1 = {a1:.3f}, b1 = {b1:.3f}")
print("D при 760 торр (без первой):", a1/760)

# Построение аппроксимирующей прямой для всех точек
x_fit = np.linspace(min(P_inv), max(P_inv), 100)
y_fit = a_all * x_fit + b_all
plt.plot(x_fit, y_fit, label='Линейная аппроксимация')
plt.grid(alpha=0.3)
plt.title("Зависимость коэффициента диффузии от обратного давления")
plt.xlabel("1/давление, 1/торр")
plt.ylabel("Коэффициент диффузии, см²/с")
plt.legend()
plt.show()

# --- Расчёт микроскопических параметров (исправлено) ---
k_B = 1.380649e-23          # Дж/К
T = 273 + 23                # К
M_He = 0.004                 # кг/моль
R = 8.314                    # Дж/(моль·К)

# Средняя тепловая скорость гелия в см/с
v_ms = np.sqrt(8 * R * T / (np.pi * M_He))   # м/с
v_cms = v_ms * 100                            # см/с

# Подготовка массивов
lambda_cm = []
sigma_cm2 = []
sigma_A2 = []

for i, p in enumerate(pressures):
    # Давление в паскалях
    P_Pa = p * 133.322
    # Концентрация фона (воздуха) в см⁻³
    n_m3 = P_Pa / (k_B * T)          # м⁻³
    n_cm3 = n_m3 * 1e-6               # см⁻³
    # Длина свободного пробега
    lam = 3 * D[i] / v_cms            # см
    lambda_cm.append(lam)
    # Сечение
    sigma = 1 / (n_cm3 * lam)         # см²
    sigma_cm2.append(sigma)
    sigma_A2.append(sigma * 1e16)     # ангстремы²

print("\nДлина свободного пробега λ, см:", lambda_cm)
print("Концентрация фона n, см⁻³:", [P_Pa/(k_B*T)*1e-6 for P_Pa in [p*133.322 for p in pressures]])
print("Сечение σ, см²:", sigma_cm2)
print("Сечение σ, Å²:", sigma_A2)