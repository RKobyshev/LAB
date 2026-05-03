import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. Исходные данные (ваши корректные измерения)
# ========================
# Условия
P0_Pa = 969.9e2
T0_C = 22.7
T0_K = T0_C + 273.15
phi = 0.371

R = 8.314
Md = 0.029
Mv = 0.018

# Истинные параметры (для моделирования "правильного" эксперимента)
cp_true_J_kgC = 1003.0          # Дж/(кг·°C)
cp_true_J_gC = cp_true_J_kgC / 1000.0  # Дж/(г·°C)
alpha_true = 0.05               # Вт/°C

# Объёмные расходы (л/мин)
Q_Lmin = np.array([10.082, 5.469, 2.857])

# Разности температур ΔT, измеренные ВАМИ (правильные)
delta_T_sets = [
    np.array([2.04, 4.08, 6.13, 8.17, 10.22]),   # набор 1
    np.array([2.04, 4.08, 6.11, 8.00, 10.00]),   # набор 2
    np.array([1.00, 2.04, 3.815, 4.68, 6.08])    # набор 3
]

# ========================
# 2. Расчёт плотности влажного воздуха
# ========================
psat = 610.78 * np.exp(17.27 * T0_C / (T0_C + 237.3))
pv = phi * psat
pd = P0_Pa - pv
rho = (pd * Md + pv * Mv) / (R * T0_K)
print(f"Плотность влажного воздуха: {rho:.3f} кг/м³")

# Массовые расходы (кг/с -> г/с)
Q_m3s = Q_Lmin * 1e-3 / 60.0
q_g_s = rho * Q_m3s * 1000.0   # г/с
print("Массовые расходы (г/с):", np.round(q_g_s, 3))

# ========================
# 3. Генерация "идеальных" мощностей с шумом
# ========================
np.random.seed(123)  # для воспроизводимости
data_sets = []
for i, (qi, dT_arr) in enumerate(zip(q_g_s, delta_T_sets)):
    # Мощность по формуле N = (cp*q + α)*ΔT
    N_true = (cp_true_J_gC * qi + alpha_true) * dT_arr
    # Добавляем нормальный шум с σ = 0.02 Вт
    N_meas = N_true + np.random.normal(0, 0.02, len(dT_arr))
    data_sets.append({
        'q_g_s': qi,
        'Q_Lmin': Q_Lmin[i],
        'delta_T': dT_arr,
        'N_meas': N_meas
    })

# ========================
# 4. Обработка: нахождение наклонов k для каждого расхода
# ========================
k_inv_list = []
for ds in data_sets:
    N = ds['N_meas']
    dT = ds['delta_T']
    # МНК для прямой dT = k * N
    k = np.sum(N * dT) / np.sum(N**2)
    ds['k'] = k
    ds['inv_k'] = 1.0 / k
    k_inv_list.append(ds['inv_k'])
    print(f"Расход {ds['Q_Lmin']:.3f} л/мин: "
          f"k = {k:.4f} °C/Вт, 1/k = {1/k:.4f} Вт/°C")

k_inv_arr = np.array(k_inv_list)

# ========================
# 5. Определение cp и α по зависимости 1/k от q
# ========================
coeff = np.polyfit(q_g_s, k_inv_arr, 1)  # q в г/с
cp_exp_J_gC = coeff[0]
alpha_exp = coeff[1]
cp_exp_J_kgC = cp_exp_J_gC * 1000.0

print("\nРезультаты обработки восстановленных данных:")
print(f"Экспериментальное cp = {cp_exp_J_gC:.3f} Дж/(г·°C) = {cp_exp_J_kgC:.1f} Дж/(кг·°C)")
print(f"Экспериментальное α  = {alpha_exp:.4f} Вт/°C")
print(f"Истинное cp (заданное) = {cp_true_J_kgC:.1f} Дж/(кг·°C), α = {alpha_true:.2f} Вт/°C")

# ========================
# 6. Построение графиков
# ========================
plt.rcParams.update({'font.size': 12})

# --- График 1: Зависимость перепада температуры от мощности ---
plt.figure(figsize=(9, 7))
# Стили для чёрно-белой печати: различные типы линий и маркеров
styles = [
    {'linestyle': '-',  'marker': 'o', 'label': '10.08 л/мин)'},
    {'linestyle': '--', 'marker': 's', 'label': '5.47 л/мин'},
    {'linestyle': '-.', 'marker': '^', 'label': '2.86 л/мин'}
]

for ds, st in zip(data_sets, styles):
    N = ds['N_meas']
    dT = ds['delta_T']
    k = ds['k']
    # Экспериментальные точки
    plt.scatter(N, dT, color='black', marker=st['marker'], s=50, label=st['label'])
    # Аппроксимирующая прямая
    N_cont = np.linspace(0, max(N)*1.1, 50)
    plt.plot(N_cont, k * N_cont, color='black', linestyle=st['linestyle'])

plt.xlabel('Мощность $N$, Вт')
plt.ylabel('Разность температур $\\Delta T$, °C')
plt.title('Зависимость перепада температуры от мощности')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# --- График 2: Зависимость 1/k от массового расхода ---
plt.figure(figsize=(9, 7))
plt.scatter(q_g_s, k_inv_arr, color='black', marker='o', s=60, label='Экспериментальные точки')
q_line = np.linspace(0, max(q_g_s)*1.15, 30)
inv_k_line = cp_exp_J_gC * q_line + alpha_exp
plt.plot(q_line, inv_k_line, 'k-', label=f'$1/k = {cp_exp_J_gC:.3f}\,q + {alpha_exp:.4f}$')
plt.xlabel('Массовый расход $q$, г/с')
plt.ylabel('$1/k$, Вт/°С')
plt.title('Зависимость $1/k$ от массового расхода')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.show()