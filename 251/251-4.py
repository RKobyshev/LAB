import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Исходные данные и предварительные расчёты
# ------------------------------------------------------------
p0 = 265.4
p01 = 417.3
pgh = p01 - p0
p_meas = [417.3, 415.6, 413.7, 411.7, 407.8, 405.9, 402.0, 398.1, 392.2]
P = [x - pgh for x in p_meas]                         # Лапласовское давление, Па
t = [21.1, 25.3, 30.2, 35.3, 40.2, 45.2, 50.0, 55.2, 60.2]
T = np.array([x + 273.15 for x in t])                 # K
R = 0.00055                                             # м
sigma = np.array([(x * R) / 2 for x in P])            # Н/м
sigma_err = 0.05 * sigma
T_err = 0.1 * np.ones_like(T)

# ------------------------------------------------------------
# 2. Линейная аппроксимация σ(T) и её параметры
# ------------------------------------------------------------
coeffs, cov = np.polyfit(T, sigma, 1, cov=True)
a, b = coeffs                     # a = dσ/dT, b = σ(0) (экстраполяция)
da = np.sqrt(cov[0, 0])

# ------------------------------------------------------------
# 3. Расчёт производных величин и их погрешностей
# ------------------------------------------------------------
q = -T * a                         # теплота, Дж/м²
U = sigma - T * a                   # полная поверхностная энергия, Дж/м²

dq = np.sqrt((a * T_err)**2 + (T * da)**2)
dU = np.sqrt(sigma_err**2 + (a * T_err)**2 + (T * da)**2)

# ------------------------------------------------------------
# 4. Подгонка степенной зависимости U = C * T^alpha (логарифмический масштаб)
# ------------------------------------------------------------
logT = np.log(T)
logU = np.log(U)   # U в Дж/м²
coeffs_log, cov_log = np.polyfit(logT, logU, 1, cov=True)
alpha, logC = coeffs_log
d_alpha = np.sqrt(cov_log[0, 0])
C = np.exp(logC)

# ------------------------------------------------------------
# 5. Построение графиков
# ------------------------------------------------------------

# График 1: σ(T)
plt.figure(1, figsize=(8, 6))
plt.errorbar(T, sigma*1000, yerr=sigma_err*1000, xerr=T_err,
             fmt='o', capsize=3, capthick=1, ecolor='black', markersize=5)
x_fit = np.linspace(min(T), max(T), 100)
y_fit = (a * x_fit + b) * 1000
plt.plot(x_fit, y_fit, 'r-', linewidth=2)
plt.xlabel('Температура, K')
plt.ylabel('σ, мН/м')
plt.title('Зависимость поверхностного натяжения воды от температуры')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# График 2: q(T) с прямой через (0,0)
plt.figure(2, figsize=(8, 6))
plt.errorbar(T, q*1000, yerr=dq*1000, xerr=T_err,
             fmt='bo', capsize=3, capthick=1, ecolor='black', markersize=4)
T_line = np.linspace(0, max(T)*1.05, 50)
q_line = -a * T_line * 1000
plt.plot(T_line, q_line, 'r-', linewidth=2)
plt.xlim(0, max(T)*1.05)
plt.ylim(0, max(q*1000)*1.1)
plt.xlabel('Температура, K')
plt.ylabel('q, мДж/м²')
plt.title('Теплота изотермического образования единицы поверхности')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# График 3: U/F (T) в логарифмическом масштабе с аппроксимацией
plt.figure(3, figsize=(8, 6))
plt.errorbar(T, U*1000, yerr=dU*1000, xerr=T_err,
             fmt='go', capsize=3, capthick=1, ecolor='black', markersize=4)
plt.xscale('log')
plt.yscale('log')

# Горизонтальная линия среднего значения (для сравнения)
U_mean = np.mean(U*1000)
plt.axhline(y=U_mean, color='r', linestyle='-', linewidth=1.5, alpha=0.8, label='Среднее')

# Линия степенной аппроксимации (по подогнанным параметрам)
T_fit_log = np.logspace(np.log10(min(T)), np.log10(max(T)), 50)
U_fit_log = C * T_fit_log**alpha * 1000   # перевод в мДж/м²
plt.plot(T_fit_log, U_fit_log, 'b--', linewidth=1.5, alpha=0.7, label=f'Степенная: α = {alpha:.3f}±{d_alpha:.3f}')

# Текстовые метки
plt.text(0.05, 0.95, f'Среднее U/F = {U_mean:.2f} мДж/м²',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.text(0.05, 0.85, f'α = {alpha:.3f} ± {d_alpha:.3f}',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Сетка
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xlabel('Температура, K')
plt.ylabel('U/F, мДж/м²')
plt.title('Полная поверхностная энергия единицы площади')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Вывод результатов в консоль
# ------------------------------------------------------------
print("\nРЕЗУЛЬТАТЫ ОБРАБОТКИ ДАННЫХ")
print("="*70)
print(f"Радиус иглы R = {R*1000:.2f} мм")
print(f"Гидростатическое давление ρgh = {pgh:.2f} Па")
print(f"Температурный коэффициент dσ/dT = {a:.3e} Н/(м·К)  (погрешность ±{da:.3e})")
print(f"Свободный член b = {b:.3e} Н/м")
print("\nРезультаты аппроксимации для U/F (степенная зависимость U = C·T^α):")
print(f"   α = {alpha:.4f} ± {d_alpha:.4f}")
print(f"   C = {C:.4e} Дж/(м²·K^α)")
print("\nТаблица рассчитанных величин:")
print("-"*80)
print(f"{'T, K':>8} {'T, °C':>8} {'σ, мН/м':>10} {'q, мДж/м²':>12} {'U/F, мДж/м²':>14}")
print("-"*80)
for i in range(len(T)):
    print(f"{T[i]:8.2f} {t[i]:8.1f} {sigma[i]*1000:10.3f} {q[i]*1000:12.3f} {U[i]*1000:14.3f}")
print("="*70)