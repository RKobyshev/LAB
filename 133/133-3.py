import numpy as np
import matplotlib.pyplot as plt

# Константы
eta_ref = 1.83e-5          # табличное значение вязкости воздуха, Па·с (при 23.4°C)
rho = 1.19                 # плотность воздуха при 23.4°C, кг/м³
Re_cr = 1000               # критическое число Рейнольдса
conv_Pa = 9.77             # перевод условных единиц (мм спирт.ст. с учётом поправки) в Па
k_Re = 0.345               # коэффициент для Re = k_Re * Q(л/мин) / R(м)

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
    """Преобразует данные, переводит давление в Па, вычисляет Re и критическое давление,
       возвращает результаты и параметры аппроксимации."""
    d_mm = data[n]['d']
    l_cm = data[n]['l']
    dP_str = data[n]['dP_str']
    Q = np.array(data[n]['Q'])

    # Перевод в СИ
    R = d_mm / 2000.0          # радиус, м
    l = l_cm / 100.0           # длина, м

    # Преобразование строк в числа (условные единицы) и перевод в Па
    P_usr = []
    for s in dP_str:
        a, b = s.split('*')
        P_usr.append(float(a) * float(b))
    P_usr = np.array(P_usr)
    P = P_usr * conv_Pa          # давление в Па

    # Сортировка по возрастанию P
    idx = np.argsort(P)
    P_sorted = P[idx]
    Q_sorted = Q[idx]

    # Число Рейнольдса для каждой точки
    Re = k_Re * Q_sorted / R      # формула Re = 0.345 * Q(л/мин) / R(м)

    # Критическое давление в Па (по формуле ΔP_кр = 8000 η² l / (ρ R³))
    P_cr = 8000 * eta_ref**2 * l / (rho * R**3)

    # Отбор ламинарных точек (Re < 1000)
    lam_mask = Re < 1000
    P_lam = P_sorted[lam_mask]
    Q_lam = Q_sorted[lam_mask]

    # Линейная аппроксимация по ламинарным точкам (если их ≥2)
    slope, intercept = None, None
    if len(P_lam) >= 2:
        coeffs = np.polyfit(P_lam, Q_lam, 1)
        slope, intercept = coeffs[0], coeffs[1]

    # Вычисление вязкости по угловому коэффициенту (л/мин/Па)
    eta_calc = None
    if slope is not None:
        # η = (π R⁴ * 60000) / (8 l * slope)   (slope в л/мин/Па)
        eta_calc = (np.pi * R**4 * 60000) / (8 * l * slope)

    # Поиск точки перехода
    Re_lam_last = None
    Re_turb_first = None
    P_lam_last = None
    P_turb_first = None
    Q_lam_last = None
    Q_turb_first = None
    if np.any(lam_mask):
        last_lam_idx = np.where(lam_mask)[0][-1]
        Re_lam_last = Re[last_lam_idx]
        P_lam_last = P_sorted[last_lam_idx]
        Q_lam_last = Q_sorted[last_lam_idx]
    if np.any(~lam_mask):
        first_turb_idx = np.where(~lam_mask)[0][0]
        Re_turb_first = Re[first_turb_idx]
        P_turb_first = P_sorted[first_turb_idx]
        Q_turb_first = Q_sorted[first_turb_idx]

    return {
        'n': n,
        'd_mm': d_mm,
        'l_cm': l_cm,
        'P_sorted': P_sorted,
        'Q_sorted': Q_sorted,
        'P_cr': P_cr,
        'P_lam': P_lam,
        'Q_lam': Q_lam,
        'slope': slope,
        'intercept': intercept,
        'eta_calc': eta_calc,
        'Re_lam_last': Re_lam_last,
        'Re_turb_first': Re_turb_first,
        'P_lam_last': P_lam_last,
        'P_turb_first': P_turb_first,
        'Q_lam_last': Q_lam_last,
        'Q_turb_first': Q_turb_first,
        'R': R,
        'l': l
    }

# Группировка по окнам
groups = {
    'Окно 1': [1, 2, 3, 4, 11],
    'Окно 2': [5, 6, 7, 8],
    'Окно 3': [9, 10]
}

# Словарь для хранения результатов всех наборов
results = {}

# Построение графиков
for group_name, indices in groups.items():
    # Определяем диаметр для группы (по первому элементу)
    d_group = data[indices[0]]['d']
    n_plots = len(indices)
    if n_plots <= 2:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
    elif n_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

    # Общий заголовок окна
    fig.suptitle(f'Зависимость объёмного расхода газа от разницы давлений для трубки диаметра d = {d_group} мм при различной длине l', fontsize=14)

    for i, n in enumerate(indices):
        ax = axes[i]
        res = process_set(n)
        results[n] = res

        P = res['P_sorted']          # Па
        Q = res['Q_sorted']
        P_cr = res['P_cr']
        P_lam = res['P_lam']
        Q_lam = res['Q_lam']
        slope = res['slope']
        intercept = res['intercept']
        l_cm = res['l_cm']

        # Все точки
        ax.scatter(P, Q, color='blue', label='Эксперимент', zorder=3)

        # Ламинарные точки (Re < 1000) – зелёные
        if len(P_lam) > 0:
            ax.scatter(P_lam, Q_lam, color='green', s=50, zorder=4,
                       label='Ламинарные (Re<1000)')

        # Линейная аппроксимация ламинарного участка
        if slope is not None:
            P_fit = np.linspace(0, max(P), 50)
            Q_fit = slope * P_fit + intercept
            ax.plot(P_fit, Q_fit, 'r--', label=f'Линейная аппрокс.\na={slope:.4f} л/мин/Па',
                    zorder=5)

        # Вертикальная линия теоретического P_кр
        ax.axvline(x=P_cr, color='magenta', linestyle=':', linewidth=2,
                   label=f'P_кр = {P_cr:.1f} Па', zorder=2)

        ax.set_xlabel('ΔP (Па)')
        ax.set_ylabel('Q (л/мин)')
        ax.set_title(f'l = {l_cm} см')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Скрыть пустые подграфики
    for j in range(len(indices), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

# Вывод результатов в консоль
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
print("="*80)

eta_values = []

for n in sorted(results.keys()):
    res = results[n]
    print(f"\nНабор {n}: d = {res['d_mm']} мм, l = {res['l_cm']} см")
    if res['slope'] is not None:
        print(f"  Угловой коэффициент a = {res['slope']:.6f} л/мин/Па")
    else:
        print("  Угловой коэффициент a = не определён")
    if res['eta_calc']:
        print(f"  Вязкость η (вычисленная) = {res['eta_calc']:.3e} Па·с")
        eta_values.append(res['eta_calc'])
    else:
        print("  Вязкость η не вычислена (мало ламинарных точек)")
    print(f"  Теоретическое P_кр = {res['P_cr']:.1f} Па")
    if res['P_lam_last'] is not None:
        print(f"  Последняя ламинарная точка: P = {res['P_lam_last']:.1f} Па, Q = {res['Q_lam_last']:.3f} л/мин, Re = {res['Re_lam_last']:.1f}")
    if res['P_turb_first'] is not None:
        print(f"  Первая турбулентная точка: P = {res['P_turb_first']:.1f} Па, Q = {res['Q_turb_first']:.3f} л/мин, Re = {res['Re_turb_first']:.1f}")

if eta_values:
    eta_mean = np.mean(eta_values)
    eta_std = np.std(eta_values, ddof=1) if len(eta_values) > 1 else 0
    print("\n" + "-"*40)
    print(f"Среднее значение вязкости по всем наборам: η = {eta_mean:.3e} ± {eta_std:.3e} Па·с")
    print(f"Табличное значение (при 23.4°C): η = {eta_ref:.3e} Па·с")
    print(f"Относительное отклонение: {abs(eta_mean - eta_ref)/eta_ref*100:.1f}%")
else:
    print("Нет данных для вычисления средней вязкости.")