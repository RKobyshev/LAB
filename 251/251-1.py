import matplotlib.pyplot as plt
import numpy as np

# Исходные данные (давление в Па, температура в °C)
P = [417.3, 415.6, 413.7, 411.7, 407.8, 405.9, 402.0, 398.1, 392.2]
t = [21.1, 25.3, 30.2, 35.3, 40.2, 45.2, 50.0, 55.2, 60.2]

# Перевод температуры в Кельвины
T = [x + 273.15 for x in t]

# Радиус иглы (м)
R = 0.0005

# Расчёт поверхностного натяжения sigma (Н/м) по формуле Лапласа: sigma = P * R / 2
sigma = [(p * R) / 2 for p in P]

# Погрешности
sigma_err = [0.05 * s for s in sigma]          # 5% от sigma
T_err = [0.1] * len(T)                          # погрешность температуры 0.1 К (постоянна)

# Построение графика с погрешностями
plt.figure(figsize=(8, 5))
plt.errorbar(T, sigma, yerr=sigma_err, xerr=T_err,
             fmt='o', capsize=3, capthick=1, ecolor='black',
             markersize=5, label='Экспериментальные точки')

# Линейная аппроксимация
coeffs = np.polyfit(T, sigma, 1)
a, b = coeffs
x_fit = np.linspace(min(T), max(T), 100)
y_fit = a * x_fit + b
plt.plot(x_fit, y_fit, 'r-', linewidth=2,
         label=f'Линейная аппроксимация: σ = ({a:.3e})·T + ({b:.3e})')

# Оформление
plt.xlabel('Температура, K')
plt.ylabel('Поверхностное натяжение, Н/м')
plt.title('Зависимость поверхностного натяжения воды от температуры')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()