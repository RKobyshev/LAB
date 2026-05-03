import matplotlib.pyplot as plt
import numpy as np

# Исходные данные
tForward = [20.41, 21.01, 22.01, 23.01, 24.01, 25.01, 26.00, 27.01, 28.00, 29.00,
            30.01, 31.00, 32.00, 33.01, 34.01, 35.01, 36.01, 37.00, 38.01, 39.00, 40.01]
PForward = [44.56, 45.9, 48.0, 50.81, 53.99, 58.14, 61.9, 65.09, 68.52, 72.48,
            76.8, 80.72, 86.11, 90.4, 95.9, 101.45, 106.54, 111.83, 116.97, 124.61, 130.71]

tBack = [38.00, 36.00, 33.99, 31.99, 29.99, 27.99, 25.99, 23.97]
Pback = [115.63, 104.53, 93.92, 84.22, 76.33, 68.16, 62.04, 56.27]

# Объединение данных с округлением температуры до целых градусов Цельсия
data = {}

for t, P in zip(tForward, PForward):
    t_int = int(round(t))
    if t_int in data:
        data[t_int][0] += P
        data[t_int][1] += 1
    else:
        data[t_int] = [P, 1]

for t, P in zip(tBack, Pback):
    t_int = int(round(t))
    if t_int in data:
        data[t_int][0] += P
        data[t_int][1] += 1
    else:
        data[t_int] = [P, 1]

# Средние значения и перевод в Кельвины
temps_C = sorted(data.keys())
P_avg = np.array([data[t][0] / data[t][1] for t in temps_C])
T_K = np.array([t + 273.15 for t in temps_C])

lnP = np.log(P_avg)
invT = 1.0 / T_K

# Линейная регрессия lnP от 1/T
coeffs = np.polyfit(invT, lnP, 1)
k = coeffs[0]
lnP_fit = np.polyval(coeffs, invT)

R = 8.314  # Дж/(моль·К)
L = -R * k

# Оценка погрешности регрессии
residuals = lnP - lnP_fit
std_lnP = np.std(residuals, ddof=2)
n = len(invT)
x_mean = np.mean(invT)
Sxx = np.sum((invT - x_mean) ** 2)
std_k = std_lnP / np.sqrt(Sxx)
std_L = R * std_k

# Вывод результатов
print("Объединённые данные (округлённая температура, среднее давление):")
for i in range(len(temps_C)):
    print(f"t = {temps_C[i]:2d} °C, T = {T_K[i]:.2f} K, P = {P_avg[i]:.2f} мм рт.ст.")

print(f"\nУгловой коэффициент k = {k:.4f}")
print(f"Теплота испарения L = {L:.1f} Дж/моль")
print(f"Оценка погрешности L: ±{std_L:.0f} Дж/моль")

# Вычисление погрешностей для графиков
delta_T = 0.02       # K
delta_P = 0.02       # мм рт.ст.
# Для первого графика: xerr = delta_T, yerr = delta_P
# Для второго: xerr = (1/T^2) * delta_T, yerr = delta_P / P
invT_err = delta_T / (T_K ** 2)   # приближённо Δ(1/T) ≈ ΔT / T^2
lnP_err = delta_P / P_avg         # Δ(ln P) = ΔP / P

# График 1: P от T
plt.figure(1)
plt.errorbar(T_K, P_avg, xerr=delta_T, yerr=delta_P,
             fmt='o', capsize=3, color='red', label='Эксперимент')
plt.plot(T_K, P_avg, 'r-')   # соединительная линия
plt.xlabel('Температура T, K')
plt.ylabel('Давление P, мм рт.ст.')
plt.title('Зависимость давления насыщенного пара от температуры')
plt.grid(True)
plt.legend()

# График 2: ln P от 1/T с погрешностями
plt.figure(2)
plt.errorbar(invT, lnP, xerr=invT_err, yerr=lnP_err,
             fmt='o', capsize=3, color='blue', label='Эксперимент')
plt.plot(invT, lnP_fit, 'r-',
         label=f'Линейная аппроксимация\nL = {L:.0f} ± {std_L:.0f} Дж/моль')
plt.xlabel('1/T, K⁻¹')
plt.ylabel('ln P')
plt.title('Зависимость ln P от обратной температуры')
plt.grid(True)
plt.legend()

plt.show()