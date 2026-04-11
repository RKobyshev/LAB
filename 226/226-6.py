import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== Текстовые константы ====================
TITLE_STD = "По стандартной формуле Стокса для всех точек"
TITLE_CORR = "По исправленной формуле Стокса для всех точек"
TITLE_COMB = "По исправленной формуле для стекла и стандартной для стали"
XLABEL = "1/T, K⁻¹"
YLABEL = "ln(η), η в Па·с"
LEGEND_EXP = "Эксперимент"
LEGEND_FIT = "Аппроксимация"
PLOT_ETA_R_FIG_TITLE = "Зависимость вязкости от радиуса шарика"
PLOT_ETA_R_XLABEL = "Радиус шарика, мм"
PLOT_ETA_R_YLABEL = "Вязкость, Па·с"
PLOT_ETA_R_LEGEND_NO_CORR = "без поправки"
PLOT_ETA_R_LEGEND_CORR = "с поправкой на стенки"

# ==================== Физические константы и параметры установки ====================
p_g = 2.5 * 1000      # кг/м³ (стекло)
p_s = 7.8 * 1000      # кг/м³ (сталь)
g = 9.815             # м/с²
l1 = 100 / 1000       # м (первый участок, не используется)
l2 = 103 / 1000       # м (второй участок, установившееся движение)

TC = [20, 25, 35, 50, 60]
T_kelvin = [i + 273.15 for i in TC]   # для построения графика 1/T

# ==================== Данные по шарикам ====================
# Диаметры (мм) – порядок: стекло, стекло, сталь, сталь
d20 = [2.1, 2.1, 0.7, 0.6]
r20 = [i / 2000 for i in d20]          # м
t2_20 = [(44.95, p_g), (45.8, p_g), (60 + 15.21, p_s), (60 + 42.19, p_s)]

d25 = [2.1, 2.1, 0.8, 0.55]
r25 = [i / 2000 for i in d25]
t2_25 = [(30.42, p_g), (29.67, p_g), (43.26, p_s), (60 + 15.31, p_s)]

d35 = [2.1, 2.1, 0.85, 0.8]
r35 = [i / 2000 for i in d35]
t2_35 = [(13.99, p_g), (13.85, p_g), (16.79, p_s), (20.52, p_s)]

d50 = [2.1, 2.1, 0.7, 0.85]
r50 = [i / 2000 for i in d50]
t2_50 = [(5.76, p_g), (5.54, p_g), (9.37, p_s), (6.47, p_s)]

d60 = [2.05, 2.1, 0.85, 0.9]
r60 = [i / 2000 for i in d60]
t2_60 = [(3.17, p_g), (3.17, p_g), (3.42, p_s), (3.46, p_s)]

# ==================== Плотность глицерина (кг/м³) ====================
p_glic_20 = 1.26 * 1000
dpdt_glic = (1.25*1000 - 1.26*1000) / (43 - 20)   # -0.4348 кг/(м³·°C)
p_glic_25 = p_glic_20 + dpdt_glic * 5
p_glic_35 = p_glic_20 + dpdt_glic * 15
p_glic_50 = p_glic_20 + dpdt_glic * 30
p_glic_60 = p_glic_20 + dpdt_glic * 40

# ==================== Скорость установившегося движения ====================
v_ust_20 = [l2 / t[0] for t in t2_20]
v_ust_25 = [l2 / t[0] for t in t2_25]
v_ust_35 = [l2 / t[0] for t in t2_35]
v_ust_50 = [l2 / t[0] for t in t2_50]
v_ust_60 = [l2 / t[0] for t in t2_60]

# ==================== Вязкость (без поправки и с поправкой на стенки) ====================
R_vessel = 0.025   # радиус сосуда, м

def eta_calc(r, v, rho_ball, rho_fluid):
    return (2/9) * g * r**2 * (rho_ball - rho_fluid) / v

# Стандартная (без поправки) и с поправкой для всех шариков
n20 = [eta_calc(r20[i], v_ust_20[i], t2_20[i][1], p_glic_20) for i in range(4)]
nS20 = [eta_calc(r20[i], v_ust_20[i] * (1 + 2.4*(r20[i]/R_vessel)), t2_20[i][1], p_glic_20) for i in range(4)]

n25 = [eta_calc(r25[i], v_ust_25[i], t2_25[i][1], p_glic_25) for i in range(4)]
nS25 = [eta_calc(r25[i], v_ust_25[i] * (1 + 2.4*(r25[i]/R_vessel)), t2_25[i][1], p_glic_25) for i in range(4)]

n35 = [eta_calc(r35[i], v_ust_35[i], t2_35[i][1], p_glic_35) for i in range(4)]
nS35 = [eta_calc(r35[i], v_ust_35[i] * (1 + 2.4*(r35[i]/R_vessel)), t2_35[i][1], p_glic_35) for i in range(4)]

n50 = [eta_calc(r50[i], v_ust_50[i], t2_50[i][1], p_glic_50) for i in range(4)]
nS50 = [eta_calc(r50[i], v_ust_50[i] * (1 + 2.4*(r50[i]/R_vessel)), t2_50[i][1], p_glic_50) for i in range(4)]

n60 = [eta_calc(r60[i], v_ust_60[i], t2_60[i][1], p_glic_60) for i in range(4)]
nS60 = [eta_calc(r60[i], v_ust_60[i] * (1 + 2.4*(r60[i]/R_vessel)), t2_60[i][1], p_glic_60) for i in range(4)]

# ==================== Комбинированная вязкость: стекло (поправка), сталь (стандарт) ====================
# Для каждой температуры: первые два (стекло) из nS, последние два (сталь) из n
n_comb20 = [nS20[0], nS20[1], n20[2], n20[3]]
n_comb25 = [nS25[0], nS25[1], n25[2], n25[3]]
n_comb35 = [nS35[0], nS35[1], n35[2], n35[3]]
n_comb50 = [nS50[0], nS50[1], n50[2], n50[3]]
n_comb60 = [nS60[0], nS60[1], n60[2], n60[3]]

# ==================== График η(r) для каждой температуры (отдельное окно) ====================
def plot_eta_vs_r():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    temps = [20, 25, 35, 50, 60]
    rad_list = [r20, r25, r35, r50, r60]
    eta_list = [n20, n25, n35, n50, n60]
    eta_corr_list = [nS20, nS25, nS35, nS50, nS60]
    p_glic_list = [p_glic_20, p_glic_25, p_glic_35, p_glic_50, p_glic_60]

    for idx, T in enumerate(temps):
        ax = axes[idx]
        rad_mm = [r * 1000 for r in rad_list[idx]]
        ax.plot(rad_mm, eta_list[idx], 'o-', label=PLOT_ETA_R_LEGEND_NO_CORR)
        ax.plot(rad_mm, eta_corr_list[idx], 's-', label=PLOT_ETA_R_LEGEND_CORR)
        ax.set_xlabel(PLOT_ETA_R_XLABEL)
        ax.set_ylabel(PLOT_ETA_R_YLABEL)
        ax.set_title(f'T = {T}°C, ρ_глиц = {p_glic_list[idx]/1000:.3f} г/см³')
        ax.legend()
        ax.grid(True)
    axes[5].axis('off')
    fig.suptitle(PLOT_ETA_R_FIG_TITLE, fontsize=14)
    plt.tight_layout()
    plt.show()

plot_eta_vs_r()

# ==================== Расчёт времени релаксации, пути, числа Re ====================
def calc_extra_params(r, v, rho_ball, rho_fluid, eta):
    tau = (2 * rho_ball * r**2) / (9 * eta)
    S_rel = v * tau
    Re = rho_fluid * v * r / eta
    return tau, S_rel, Re

results = {}
temps_data = {
    20: (r20, t2_20, v_ust_20, n20, p_glic_20),
    25: (r25, t2_25, v_ust_25, n25, p_glic_25),
    35: (r35, t2_35, v_ust_35, n35, p_glic_35),
    50: (r50, t2_50, v_ust_50, n50, p_glic_50),
    60: (r60, t2_60, v_ust_60, n60, p_glic_60),
}
for temp, (radii, t2_list, v_list, eta_list, rho_fluid) in temps_data.items():
    temp_results = []
    for i in range(4):
        r = radii[i]
        v = v_list[i]
        rho_ball = t2_list[i][1]
        eta = eta_list[i]
        tau, S_rel, Re = calc_extra_params(r, v, rho_ball, rho_fluid, eta)
        temp_results.append({
            'шарик_№': i+1,
            'материал': 'стекло' if rho_ball == p_g else 'сталь',
            'r_мм': r*1000,
            'v_м/с': v,
            'eta_Па·с': eta,
            'tau_с': tau,
            'S_рел_мм': S_rel*1000,
            'Re': Re
        })
    results[temp] = temp_results

# Вывод основной таблицы (включая времена установления)
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ (стандартная формула для всех шариков)")
print("="*80)
for temp, recs in results.items():
    print(f"\nТемпература {temp}°C")
    print(f"{'№':2} {'Материал':6} {'r, мм':6} {'v, м/с':8} {'η, Па·с':10} {'τ, с':8} {'S_rel, мм':9} {'Re':8}")
    for rec in recs:
        print(f"{rec['шарик_№']:2} {rec['материал']:6} {rec['r_мм']:6.3f} {rec['v_м/с']:8.4f} {rec['eta_Па·с']:10.4f} {rec['tau_с']:8.4f} {rec['S_рел_мм']:9.2f} {rec['Re']:8.4f}")

# Отдельный вывод времени установления (для наглядности)
print("\n" + "="*80)
print("ВРЕМЯ РЕЛАКСАЦИИ И ПУТЬ УСТАНОВЛЕНИЯ (стандартная формула)")
print("="*80)
for temp, recs in results.items():
    print(f"\nТемпература {temp}°C:")
    for rec in recs:
        print(f"  Шарик {rec['шарик_№']} ({rec['материал']}): τ = {rec['tau_с']:.4f} с, S_рел = {rec['S_рел_мм']:.2f} мм")

# ==================== Оценка погрешностей ====================
dt = 0.5               # с
dL = 0.0               # погрешность длины (пренебрегаем)
dr_glass = 0.05 / 1000 # м
dr_steel = 0.1 / 1000  # м

def calc_eta_with_error(r, t, rho_ball, rho_fluid, l, dt, dr, dL):
    v = l / t
    eps_v = np.sqrt((dL/l)**2 + (dt/t)**2)
    eps_r = dr / r
    eps_eta = np.sqrt((2*eps_r)**2 + eps_v**2)
    eta = (2/9) * g * r**2 * (rho_ball - rho_fluid) / v
    d_eta = eta * eps_eta
    return eta, d_eta

temps_data_full = {
    20: (r20, t2_20, p_glic_20, (dr_glass, dr_steel)),
    25: (r25, t2_25, p_glic_25, (dr_glass, dr_steel)),
    35: (r35, t2_35, p_glic_35, (dr_glass, dr_steel)),
    50: (r50, t2_50, p_glic_50, (dr_glass, dr_steel)),
    60: (r60, t2_60, p_glic_60, (dr_glass, dr_steel)),
}

# Для стандартной формулы (все без поправки)
eta_std_with_err = {}
# Для поправленной (все с поправкой)
eta_corr_with_err = {}
# Для комбинированной (стекло с поправкой, сталь стандарт)
eta_comb_with_err = {}

for T, (radii, t_list, rho_f, (dr_g, dr_s)) in temps_data_full.items():
    eta_list_std = []
    eta_list_corr = []
    eta_list_comb = []
    for i in range(4):
        r = radii[i]
        t = t_list[i][0]
        rho_b = t_list[i][1]
        dr = dr_g if rho_b == p_g else dr_s
        # Стандартная формула
        eta_std, d_eta_std = calc_eta_with_error(r, t, rho_b, rho_f, l2, dt, dr, dL)
        eta_list_std.append((eta_std, d_eta_std))
        # Формула с поправкой
        corr_factor = 1 + 2.4 * (r / R_vessel)
        eta_corr = eta_std / corr_factor
        d_eta_corr = d_eta_std / corr_factor
        eta_list_corr.append((eta_corr, d_eta_corr))
        # Комбинированная: для стекла (i=0,1) берём с поправкой, для стали (i=2,3) стандартную
        if i < 2:  # стекло
            eta_list_comb.append((eta_corr, d_eta_corr))
        else:      # сталь
            eta_list_comb.append((eta_std, d_eta_std))
    eta_std_with_err[T] = eta_list_std
    eta_corr_with_err[T] = eta_list_corr
    eta_comb_with_err[T] = eta_list_comb

def mean_and_error(values_with_err):
    vals = [v[0] for v in values_with_err]
    errs = [v[1] for v in values_with_err]
    mean_val = np.mean(vals)
    mean_err = np.sqrt(np.sum(np.square(errs))) / len(vals)
    return mean_val, mean_err

# Вычисляем средние для каждого типа
eta_mean_std = []
eta_mean_std_err = []
eta_mean_corr = []
eta_mean_corr_err = []
eta_mean_comb = []
eta_mean_comb_err = []
sorted_T = sorted(temps_data_full.keys())
for T in sorted_T:
    m_std, e_std = mean_and_error(eta_std_with_err[T])
    m_corr, e_corr = mean_and_error(eta_corr_with_err[T])
    m_comb, e_comb = mean_and_error(eta_comb_with_err[T])
    eta_mean_std.append(m_std)
    eta_mean_std_err.append(e_std)
    eta_mean_corr.append(m_corr)
    eta_mean_corr_err.append(e_corr)
    eta_mean_comb.append(m_comb)
    eta_mean_comb_err.append(e_comb)

# Преобразуем в массивы NumPy
eta_mean_std = np.array(eta_mean_std)
eta_mean_std_err = np.array(eta_mean_std_err)
eta_mean_corr = np.array(eta_mean_corr)
eta_mean_corr_err = np.array(eta_mean_corr_err)
eta_mean_comb = np.array(eta_mean_comb)
eta_mean_comb_err = np.array(eta_mean_comb_err)

# ln(η) и погрешность
ln_eta_std = np.log(eta_mean_std)
d_ln_eta_std = eta_mean_std_err / eta_mean_std
ln_eta_corr = np.log(eta_mean_corr)
d_ln_eta_corr = eta_mean_corr_err / eta_mean_corr
ln_eta_comb = np.log(eta_mean_comb)
d_ln_eta_comb = eta_mean_comb_err / eta_mean_comb

# 1/T
T_kelvin_sorted = np.array([t + 273.15 for t in sorted_T])
invT = 1.0 / T_kelvin_sorted

# ==================== Взвешенная линейная регрессия ====================
def weighted_linreg(x, y, yerr):
    w = 1.0 / np.square(yerr)
    xw = np.sum(w * x) / np.sum(w)
    yw = np.sum(w * y) / np.sum(w)
    num = np.sum(w * (x - xw) * (y - yw))
    den = np.sum(w * (x - xw)**2)
    slope = num / den
    intercept = yw - slope * xw
    slope_err = np.sqrt(1.0 / den)
    return slope, intercept, slope_err

slope_std, intercept_std, slope_std_err = weighted_linreg(invT, ln_eta_std, d_ln_eta_std)
slope_corr, intercept_corr, slope_corr_err = weighted_linreg(invT, ln_eta_corr, d_ln_eta_corr)
slope_comb, intercept_comb, slope_comb_err = weighted_linreg(invT, ln_eta_comb, d_ln_eta_comb)

# ==================== Энергия активации ====================
k_B = 1.380649e-23
W_std_J = k_B * slope_std
W_std_err_J = k_B * slope_std_err
W_corr_J = k_B * slope_corr
W_corr_err_J = k_B * slope_corr_err
W_comb_J = k_B * slope_comb
W_comb_err_J = k_B * slope_comb_err

N_A = 6.02214076e23
eV = 1.60217662e-19
def to_kJmol(W_J): return W_J * N_A / 1000
def to_eV(W_J): return W_J / eV

# ==================== Построение трёх отдельных графиков ====================
x_fit = np.linspace(min(invT), max(invT), 100)

# График 1: стандартная формула (все без поправки)
plt.figure(figsize=(6, 5))
plt.errorbar(invT, ln_eta_std, yerr=d_ln_eta_std, fmt='o', capsize=5, label=LEGEND_EXP)
plt.plot(x_fit, slope_std*x_fit + intercept_std, 'r--',
         label=f'{LEGEND_FIT} (наклон = {slope_std:.0f}±{slope_std_err:.0f} K)')
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)
plt.title(TITLE_STD)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График 2: с поправкой для всех шариков
plt.figure(figsize=(6, 5))
plt.errorbar(invT, ln_eta_corr, yerr=d_ln_eta_corr, fmt='s', capsize=5, label=LEGEND_EXP)
plt.plot(x_fit, slope_corr*x_fit + intercept_corr, 'r--',
         label=f'{LEGEND_FIT} (наклон = {slope_corr:.0f}±{slope_corr_err:.0f} K)')
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)
plt.title(TITLE_CORR)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График 3: комбинированный (стекло с поправкой, сталь стандарт)
plt.figure(figsize=(6, 5))
plt.errorbar(invT, ln_eta_comb, yerr=d_ln_eta_comb, fmt='^', capsize=5, label=LEGEND_EXP)
plt.plot(x_fit, slope_comb*x_fit + intercept_comb, 'r--',
         label=f'{LEGEND_FIT} (наклон = {slope_comb:.0f}±{slope_comb_err:.0f} K)')
plt.xlabel(XLABEL)
plt.ylabel(YLABEL)
plt.title(TITLE_COMB)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==================== Вывод результатов энергии активации ====================
print("\n" + "="*80)
print("ЭНЕРГИЯ АКТИВАЦИИ (по трём вариантам)")
print("="*80)
print(f"\n1. Стандартная формула (все шарики без поправки):")
print(f"   Наклон (W/k) = {slope_std:.0f} ± {slope_std_err:.0f} K")
print(f"   W = {W_std_J:.2e} ± {W_std_err_J:.2e} Дж")
print(f"     = {to_eV(W_std_J):.3f} ± {to_eV(W_std_err_J):.3f} эВ")
print(f"     = {to_kJmol(W_std_J):.1f} ± {to_kJmol(W_std_err_J):.1f} кДж/моль")

print(f"\n2. С поправкой на стенки (все шарики):")
print(f"   Наклон (W/k) = {slope_corr:.0f} ± {slope_corr_err:.0f} K")
print(f"   W = {W_corr_J:.2e} ± {W_corr_err_J:.2e} Дж")
print(f"     = {to_eV(W_corr_J):.3f} ± {to_eV(W_corr_err_J):.3f} эВ")
print(f"     = {to_kJmol(W_corr_J):.1f} ± {to_kJmol(W_corr_err_J):.1f} кДж/моль")

print(f"\n3. Комбинированный вариант (стекло – поправка, сталь – стандарт):")
print(f"   Наклон (W/k) = {slope_comb:.0f} ± {slope_comb_err:.0f} K")
print(f"   W = {W_comb_J:.2e} ± {W_comb_err_J:.2e} Дж")
print(f"     = {to_eV(W_comb_J):.3f} ± {to_eV(W_comb_err_J):.3f} эВ")
print(f"     = {to_kJmol(W_comb_J):.1f} ± {to_kJmol(W_comb_err_J):.1f} кДж/моль")