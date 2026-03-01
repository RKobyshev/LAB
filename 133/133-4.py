import numpy as np
import matplotlib.pyplot as plt

# Константы
eta_ref = 1.83e-5          # табличное значение вязкости воздуха, Па·с (при 23.4°C)
rho = 1.19                 # плотность воздуха при 23.4°C, кг/м³
Re_cr = 1000               # критическое число Рейнольдса
conv_Pa = 9.77             # перевод условных единиц (мм спирт.ст. с учётом поправки) в Па
k_Re = 0.345               # коэффициент для Re = k_Re * Q(л/мин) / R(м)

# Погрешности
sigma_d = {3.95: 0.05, 5.30: 0.05, 3.00: 0.1}  # мм
sigma_l = 0.5              # см
sigma_rho_rel = 0.01
sigma_eta_tabl_rel = 0.01

# Данные измерений
data = {
    1: {'d': 3.95, 'l': 50,
        'dP_str': ['10*0.2', '60*0.2', '110*0.2', '160*0.2', '210*0.2', '260*0.2', '200*0.4'],
        'Q': [0.7, 4.350, 6.2, 7.1, 8.1, 9.1, 11]},
    2: {'d': 3.95, 'l': 90,
        'dP_str': ['10*0.4', '60*0.4', '110*0.4', '160*0.4', '210*0.4', '260*0.4', '200*0.6', '250*0.6'],
        'Q': [0.63, 4.4, 6.2, 7.08, 8.2, 9.2, 9.98, 11.21]},
    3: {'d': 3.95, 'l': 100,
        'dP_str': ['10*0.4', '60*0.4'],
        'Q': [0.269, 3.6]},
    4: {'d': 3.95, 'l': 30,
        'dP_str': ['10*0.4', '60*0.4', '110*0.4', '35*0.4', '85*0.4'],
        'Q': [2.550, 8.116, 11.224, 6.320, 9.724]},
    5: {'d': 5.30, 'l': 30,
        'dP_str': ['10*0.4', '30*0.4', '80*0.2', '100*0.2', '110*0.2'],
        'Q': [7.050, 13.450, 13.956, 15.605, 16.377]},
    6: {'d': 5.30, 'l': 70,
        'dP_str': ['10*0.2', '40*0.2', '70*0.2', '100*0.2', '130*0.2', '160*0.2', '190*0.2', '220*0.2', '240*0.2', '270*0.2'],
        'Q': [1.632, 6.253, 8.672, 9.883, 11.124, 12.466, 13.864, 14.835, 15.447, 16.454]},
    7: {'d': 5.30, 'l': 90,
        'dP_str': ['10*0.2', '40*0.2', '70*0.2', '100*0.2', '130*0.2', '160*0.2', '190*0.2'],
        'Q': [1.421, 5.463, 8.133, 9.039, 9.770, 10.894, 11.965]},
    8: {'d': 5.30, 'l': 50,
        'dP_str': ['10*0.2', '40*0.2', '70*0.2', '110*0.2', '150*0.2'],
        'Q': [2.391, 8.320, 9.848, 12.311, 14.659]},
    9: {'d': 3.00, 'l': 20,
        'dP_str': ['5*0.2', '10*0.2', '14*0.2', '20*0.2', '25*0.2', '30*0.2'],
        'Q': [1.043, 1.905, 2.560, 3.159, 3.808, 4.334]},
    10: {'d': 3.00, 'l': 50,
         'dP_str': ['5*0.2', '10*0.2', '20*0.2', '30*0.2', '40*0.2', '49*0.2', '60*0.2', '70*0.2'],
         'Q': [0.440, 0.756, 1.721, 2.343, 2.952, 3.582, 4.070, 4.677]},
    11: {'d': 3.95, 'l': 40,
         'dP_str': ['10*0.4', '60*0.4', '110*0.4', '160*0.4'],
         'Q': [1.853, 6.556, 8.455, 10.596]}
}

def process_set(n):
    d_mm = data[n]['d']
    l_cm = data[n]['l']
    dP_str = data[n]['dP_str']
    Q = np.array(data[n]['Q'])

    # Перевод в СИ
    R = d_mm / 2000.0
    l = l_cm / 100.0

    # Преобразование строк в числа (условные единицы) и перевод в Па
    P_usr = []
    for s in dP_str:
        a, b = s.split('*')
        P_usr.append(float(a) * float(b))
    P_usr = np.array(P_usr)
    P = P_usr * conv_Pa

    # Сортировка
    idx = np.argsort(P)
    P_sorted = P[idx]
    Q_sorted = Q[idx]

    # Число Рейнольдса
    Re = k_Re * Q_sorted / R

    # Погрешности измерений
    sigma_P_usr = 0.5 * (np.array([float(s.split('*')[1]) for s in dP_str]))  # 0.5*K
    sigma_P = sigma_P_usr * conv_Pa
    sigma_Q = np.array([0.0005 if abs(q - round(q,3)) < 1e-10 else 0.05 for q in Q_sorted])

    # Относительные погрешности
    eps_P = sigma_P / P_sorted
    eps_Q = sigma_Q / Q_sorted

    # Критическое давление (теор)
    sigma_R = sigma_d[d_mm] / 2000.0
    eps_R = sigma_R / R
    eps_l = sigma_l / l_cm
    P_cr = 8 * eta_ref**2 * 1000 * l / (rho * R**3)
    eps_Pcr = np.sqrt((2*sigma_eta_tabl_rel)**2 + eps_l**2 + sigma_rho_rel**2 + (3*eps_R)**2)
    sigma_Pcr = P_cr * eps_Pcr

    # Отбор ламинарных точек
    lam_mask = Re < 1000
    P_lam = P_sorted[lam_mask]
    Q_lam = Q_sorted[lam_mask]

    # Если есть хотя бы одна ламинарная точка, вычисляем eta
    eta_calc = None
    sigma_eta = None
    b = None
    sigma_b = None

    if np.sum(lam_mask) >= 2:
        # Используем две крайние ламинарные точки
        P1, P2 = P_lam[0], P_lam[-1]
        Q1, Q2 = Q_lam[0], Q_lam[-1]
        a = (Q2 - Q1) / (P2 - P1)
        # Погрешность a
        sigma_dQ = np.sqrt(sigma_Q[lam_mask][0]**2 + sigma_Q[lam_mask][-1]**2)
        sigma_dP = np.sqrt(sigma_P[lam_mask][0]**2 + sigma_P[lam_mask][-1]**2)
        eps_a = np.sqrt((sigma_dQ/(Q2-Q1))**2 + (sigma_dP/(P2-P1))**2)
        sigma_a = a * eps_a
        b = 1/a
        eps_b = eps_a
        sigma_b = b * eps_b
        # Вязкость
        eta_calc = np.pi * R**4 * 60000 / (8 * l * a)
        eps_eta = np.sqrt((4*eps_R)**2 + eps_l**2 + eps_a**2)
        sigma_eta = eta_calc * eps_eta
    elif np.sum(lam_mask) == 1:
        # Единственная ламинарная точка
        P_pt = P_lam[0]
        Q_pt = Q_lam[0]
        b = P_pt / Q_pt
        eps_b = np.sqrt(eps_P[lam_mask][0]**2 + eps_Q[lam_mask][0]**2)
        sigma_b = b * eps_b
        eta_calc = np.pi * R**4 * 60000 * P_pt / (8 * l * Q_pt)
        eps_eta = np.sqrt((4*eps_R)**2 + eps_l**2 + eps_P[lam_mask][0]**2 + eps_Q[lam_mask][0]**2)
        sigma_eta = eta_calc * eps_eta
    # else: нет ламинарных точек — оставляем None

    # Последняя ламинарная точка (если есть)
    last_lam = None
    if np.sum(lam_mask) > 0:
        last_idx = np.where(lam_mask)[0][-1]
        last_lam = (P_sorted[last_idx], Q_sorted[last_idx], Re[last_idx],
                    sigma_P[last_idx], sigma_Q[last_idx])

    # Первая турбулентная точка (если есть)
    first_turb = None
    turb_mask = ~lam_mask
    if np.sum(turb_mask) > 0:
        first_idx = np.where(turb_mask)[0][0]
        first_turb = (P_sorted[first_idx], Q_sorted[first_idx], Re[first_idx],
                      sigma_P[first_idx], sigma_Q[first_idx])

    return {
        'n': n,
        'd_mm': d_mm,
        'l_cm': l_cm,
        'b': b,
        'sigma_b': sigma_b,
        'eta_calc': eta_calc,
        'sigma_eta': sigma_eta,
        'P_cr': P_cr,
        'sigma_Pcr': sigma_Pcr,
        'last_lam': last_lam,
        'first_turb': first_turb,
        'P_sorted': P_sorted,
        'Q_sorted': Q_sorted,
        'Re': Re,
        'sigma_P': sigma_P,
        'sigma_Q': sigma_Q
    }

# Обработка всех наборов
results = {}
for n in data:
    results[n] = process_set(n)

# Вывод таблиц
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ОБРАБОТКИ (ТАБЛИЦЫ ПО ДИАМЕТРАМ)")
print("="*80)

# Группировка по диаметрам
for diam in [3.95, 5.30, 3.00]:
    print(f"\n--- Диаметр {diam} мм ---")
    print(" l, см | b, Па/(л/мин)         | η, 10⁻⁵ Па·с          | Pкр теор, Па          | Последняя ламинарная точка")
    print("-------|------------------------|------------------------|------------------------|----------------------------------------")
    for n in sorted(results.keys()):
        res = results[n]
        if abs(res['d_mm'] - diam) < 0.01:
            l_str = f"{res['l_cm']:3d}"
            # b
            if res['b'] is not None:
                b_str = f"{res['b']:6.2f} ± {res['sigma_b']:5.2f}"
            else:
                b_str = " – "
            # eta
            if res['eta_calc'] is not None:
                eta_val = res['eta_calc']*1e5
                eta_err = res['sigma_eta']*1e5
                eta_str = f"{eta_val:5.3f} ± {eta_err:5.3f}"
            else:
                eta_str = " – "
            # Pcr
            pcr_str = f"{res['P_cr']:5.1f} ± {res['sigma_Pcr']:4.1f}"
            # last lam
            if res['last_lam']:
                P_ll, Q_ll, Re_ll, sigP, sigQ = res['last_lam']
                last_str = f"P={P_ll:5.1f}±{sigP:3.1f} Па, Q={Q_ll:5.3f}±{sigQ:.4f} л/мин, Re={Re_ll:5.1f}"
            else:
                last_str = " – "
            print(f"{l_str:>3s}   | {b_str:20s} | {eta_str:20s} | {pcr_str:20s} | {last_str}")

# Вывод дополнительной информации о переходах
print("\n" + "="*80)
print("ПЕРЕХОД К ТУРБУЛЕНТНОСТИ")
print("="*80)
for n in sorted(results.keys()):
    res = results[n]
    if res['first_turb']:
        P_ft, Q_ft, Re_ft, sigP, sigQ = res['first_turb']
        print(f"Набор {n} (d={res['d_mm']} мм, l={res['l_cm']} см): первая турбулентная точка: P={P_ft:5.1f}±{sigP:3.1f} Па, Q={Q_ft:5.3f}±{sigQ:.4f} л/мин, Re={Re_ft:5.1f}")
    else:
        print(f"Набор {n} (d={res['d_mm']} мм, l={res['l_cm']} см): турбулентных точек нет")

# Построение графика для турбулентных точек в логарифмическом масштабе с отдельными аппроксимациями
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20(np.linspace(0, 1, 11))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H']

# Словарь для хранения коэффициентов наклона
slopes = {}

for idx, n in enumerate(sorted(results.keys())):
    res = results[n]
    turb_mask = res['Re'] >= 1000
    if np.any(turb_mask):
        P_turb = res['P_sorted'][turb_mask]
        Q_turb = res['Q_sorted'][turb_mask]
        # Избегаем нулевых значений
        mask = (P_turb > 0) & (Q_turb > 0)
        P_plot = P_turb[mask]
        Q_plot = Q_turb[mask]

        if len(P_plot) >= 2:
            # Линейная аппроксимация в логарифмических координатах
            logP = np.log(P_plot)
            logQ = np.log(Q_plot)
            coeffs = np.polyfit(logP, logQ, 1)
            k = coeffs[0]
            b = coeffs[1]
            slopes[n] = (k, b, res['d_mm'], res['l_cm'])

            # Точки
            plt.loglog(P_plot, Q_plot, marker=markers[idx % len(markers)], linestyle='None',
                       color=colors[idx], markersize=6, label=f'd={res["d_mm"]} мм, l={res["l_cm"]} см, k={k:.3f}')
            # Линия регрессии
            P_fit = np.linspace(min(P_plot), max(P_plot), 50)
            logP_fit = np.log(P_fit)
            logQ_fit = k * logP_fit + b
            plt.loglog(P_fit, np.exp(logQ_fit), '--', color=colors[idx], linewidth=1.5)
        else:
            # Если точек мало, просто отображаем точки без линии
            plt.loglog(P_plot, Q_plot, marker=markers[idx % len(markers)], linestyle='None',
                       color=colors[idx], markersize=6, label=f'd={res["d_mm"]} мм, l={res["l_cm"]} см (мало точек)')
    else:
        print(f"Набор {n} (d={res['d_mm']} мм, l={res['l_cm']} см) не имеет турбулентных точек")

# Вывод коэффициентов наклона в консоль
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ АППРОКСИМАЦИИ ТУРБУЛЕНТНЫХ УЧАСТКОВ (log Q = k * log P + b)")
print("="*80)
for n in sorted(slopes.keys()):
    k, b, d, l = slopes[n]
    print(f"Набор {n} (d={d} мм, l={l} см): k = {k:.3f}, b = {b:.3f}")

plt.xlabel('ΔP (Па)')
plt.ylabel('Q (л/мин)')
plt.title('Турбулентные точки в логарифмическом масштабе\n(ожидаемая зависимость Q ∝ √ΔP соответствует наклону k = 0.5)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()