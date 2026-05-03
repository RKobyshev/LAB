import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. Параметры эксперимента
# ========================
# Атмосферные условия
P0_Pa = 969.9e2          # 969.9 гПа -> Па
T0_C = 22.7              # °C
T0_K = T0_C + 273.15     # K
phi = 0.371              # относительная влажность

# Физические константы
R = 8.314                # Дж/(моль·К)
Md = 0.029               # кг/моль (сухой воздух)
Mv = 0.018               # кг/моль (вода)

# Параметры нагревателя и измерителя
beta = 40.7e-6           # В/°C (чувствительность термопары)
cp_true = 1003.0         # Дж/(кг·°C) - истинная теплоёмкость
alpha_true = 0.05        # Вт/°C - коэффициент потерь

# Объёмные расходы (л/мин)
Q_Lmin = np.array([10.082, 5.469, 2.857])

# ========================
# 2. Расчёт плотности влажного воздуха
# ========================
# Давление насыщенного пара при T0 (упрощённая формула)
psat = 610.78 * np.exp(17.27 * T0_C / (T0_C + 237.3))   # Па
pv = phi * psat
pd = P0_Pa - pv

rho = (pd * Md + pv * Mv) / (R * T0_K)    # кг/м³
print(f"Плотность влажного воздуха: {rho:.3f} кг/м³")

# Массовые расходы (кг/с)
Q_m3s = Q_Lmin * 1e-3 / 60.0   # л/мин -> м³/с
q = rho * Q_m3s                # кг/с
print("Массовые расходы (г/с):", np.round(q * 1000, 3))

# ========================
# 3. Генерация "экспериментальных" данных
# ========================
np.random.seed(42)   # для воспроизводимости
num_points = 5

# Будем хранить данные в списках словарей
data_sets = []
for i, qi in enumerate(q):
    # Задаём равномерно распределённые ΔT от 2 до 10 °C
    delta_T = np.linspace(2.0, 10.0, num_points)
    # Истинная мощность по модели N = (cp*q + alpha)*ΔT
    N_true = (cp_true * qi + alpha_true) * delta_T
    # Добавляем небольшой нормальный шум (σ = 0.02 Вт)
    N_meas = N_true + np.random.normal(0, 0.02, num_points)
    data_sets.append({
        'q': qi,
        'Q_Lmin': Q_Lmin[i],
        'delta_T': delta_T,
        'N_meas': N_meas
    })

# ========================
# 4. Обработка: нахождение наклонов k_i
# ========================
k_vals = []
k_vals_inv = []
for ds in data_sets:
    N = ds['N_meas']
    dT = ds['delta_T']
    # МНК для модели dT = k * N  (без свободного члена)
    k = np.sum(N * dT) / np.sum(N**2)
    ds['k'] = k
    ds['inv_k'] = 1.0 / k
    k_vals.append(k)
    k_vals_inv.append(1.0 / k)
    print(f"Расход {ds['Q_Lmin']:.3f} л/мин: k = {k:.4f} °C/Вт, 1/k = {1/k:.4f} Вт/°C")

k_vals = np.array(k_vals)
k_vals_inv = np.array(k_vals_inv)

# ========================
# 5. Определение cp и α из графика 1/k vs q
# ========================
# Линейная регрессия: inv_k = cp * q + alpha
coeff = np.polyfit(q * 1000, k_vals_inv, 1)   # q в г/с для удобства
cp_exp = coeff[0]       # Дж/(г·°C) = 1000*Дж/(кг·°C)
alpha_exp = coeff[1]    # Вт/°C
cp_exp_SI = cp_exp * 1000   # Дж/(кг·°C)

print("\nРезультаты обработки:")
print(f"Экспериментальное cp = {cp_exp:.3f} Дж/(г·°C) = {cp_exp_SI:.1f} Дж/(кг·°C)")
print(f"Экспериментальное α  = {alpha_exp:.4f} Вт/°C")
print(f"Истинное cp = {cp_true} Дж/(кг·°C), α = {alpha_true} Вт/°C")

# ========================
# 6. Построение графиков
# ========================
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------- График 1: ΔT от N для трёх расходов ----------
ax1 = axes[0]
colors = ['blue', 'green', 'red']
for i, ds in enumerate(data_sets):
    N_meas = ds['N_meas']
    dT = ds['delta_T']
    k = ds['k']
    # Экспериментальные точки
    ax1.scatter(N_meas, dT, color=colors[i], label=f"q = {ds['Q_Lmin']:.2f} л/мин")
    # Аппроксимирующая прямая
    N_line = np.linspace(0, max(N_meas)*1.1, 50)
    dT_line = k * N_line
    ax1.plot(N_line, dT_line, '--', color=colors[i])

ax1.set_xlabel("Мощность N, Вт")
ax1.set_ylabel("Разность температур ΔT, °C")
ax1.set_title("Зависимость ΔT от мощности нагрева")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# ---------- График 2: 1/k от q ----------
ax2 = axes[1]
q_plot = q * 1000    # г/с
ax2.scatter(q_plot, k_vals_inv, color='black', zorder=5)
# Регрессионная прямая
q_line = np.linspace(0, max(q_plot)*1.2, 20)
inv_k_line = cp_exp * q_line + alpha_exp
ax2.plot(q_line, inv_k_line, 'r-', label=f"1/k = {cp_exp:.3f}·q + {alpha_exp:.4f}")
ax2.set_xlabel("Массовый расход q, г/с")
ax2.set_ylabel("1/k, Вт/°C")
ax2.set_title("Определение cp и α")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()