import numpy as np
import matplotlib.pyplot as plt
import math

# Константы
eta = 1.83e-5          # вязкость воздуха, Па·с
rho = 1.19             # плотность воздуха, кг/м³ (при 23.4°C)
Re_cr = 1000           # критическое число Рейнольдса
conv_Pa = 9.77         # перевод условных единиц (N*K) в Па (с учётом n=0.996)
# Коэффициент для вычисления Re по Q (л/мин) и R (м): Re = k * Q / R, где k = 0.345
k_Re = 0.345

# Данные измерений (диаметр в мм, длина в см, перепады в виде строк, расход в л/мин)
data = {
    1: {'d': 3.95, 'l': 50, 'dP_str': ['10*0.2', '60*0.2', '110*0.2', '160*0.2', '210*0.2', '260*0.2', '200*0.4'],
        'Q': [0.7, 4.350, 6.2, 7.1, 8.1, 9.1, 11]},
    2: {'d': 3.95, 'l': 90, 'dP_str': ['10*0.4', '60*0.4', '110*0.4', '160*0.4', '210*0.4', '260*0.4', '200*0.6', '250*0.6'],
        'Q': [0.63, 4.4, 6.2, 7.08, 8.2, 9.2, 9.98, 11.21]},
    3: {'d': 3.95, 'l': 100, 'dP_str': ['10*0.4', '60*0.4'], 'Q': [0.269, 3.6]},
    4: {'d': 3.95, 'l': 30, 'dP_str': ['10*0.4', '60*0.4', '110*0.4', '35*0.4', '85*0.4'],
        'Q': [2.550, 8.116, 11.224, 6.320, 9.724]},
    5: {'d': 5.30, 'l': 30, 'dP_str': ['10*0.4', '30*0.4', '80*0.2', '100*0.2', '110*0.2'],
        'Q': [7.050, 13.450, 13.956, 15.605, 16.377]},
    6: {'d': 5.30, 'l': 70, 'dP_str': ['10*0.2', '40*0.2', '70*0.2', '100*0.2', '130*0.2', '160*0.2', '190*0.2', '220*0.2', '240*0.2', '270*0.2'],
        'Q': [1.632, 6.253, 8.672, 9.883, 11.124, 12.466, 13.864, 14.835, 15.447, 16.454]},
    7: {'d': 5.30, 'l': 90, 'dP_str': ['10*0.2', '40*0.2', '70*0.2', '100*0.2', '130*0.2', '160*0.2', '190*0.2'],
        'Q': [1.421, 5.463, 8.133, 9.039, 9.770, 10.894, 11.965]},
    8: {'d': 5.30, 'l': 50, 'dP_str': ['10*0.2', '40*0.2', '70*0.2', '110*0.2', '150*0.2'],
        'Q': [2.391, 8.320, 9.848, 12.311, 14.659]},
    9: {'d': 3.00, 'l': 20, 'dP_str': ['5*0.2', '10*0.2', '14*0.2', '20*0.2', '25*0.2', '30*0.2'],
        'Q': [1.043, 1.905, 2.560, 3.159, 3.808, 4.334]},
    10: {'d': 3.00, 'l': 50, 'dP_str': ['5*0.2', '10*0.2', '20*0.2', '30*0.2', '40*0.2', '49*0.2', '60*0.2', '70*0.2'],
         'Q': [0.440, 0.756, 1.721, 2.343, 2.952, 3.582, 4.070, 4.677]},
    11: {'d': 3.95, 'l': 40, 'dP_str': ['10*0.4', '60*0.4', '110*0.4', '160*0.4'],
         'Q': [1.853, 6.556, 8.455, 10.596]}
}

def process_set(n):
    """
    Обрабатывает один набор данных.
    Возвращает:
        P_sorted, Q_sorted, P_cr, P_lam, Q_lam, slope, intercept, d, l_cm
    """
    d = data[n]['d']          # мм
    l_cm = data[n]['l']       # см
    dP_str = data[n]['dP_str']
    Q = np.array(data[n]['Q'])

    # Перевод в метры
    R = d / 2000.0            # радиус в м
    l = l_cm / 100.0          # длина в м

    # Вычисление критического давления в Па, затем перевод в условные единицы
    # ΔP_кр = (8 * η² * Re * l) / (ρ * R³)   (Па)
    P_cr_Pa = (8 * eta**2 * Re_cr * l) / (rho * R**3)
    P_cr = P_cr_Pa / conv_Pa   # условные единицы

    # Преобразование dP_str в числа
    P_vals = []
    for s in dP_str:
        a, b = s.split('*')
        P_vals.append(float(a) * float(b))
    P_vals = np.array(P_vals)

    # Сортировка по возрастанию P (обязательно для анализа)
    sort_idx = np.argsort(P_vals)
    P_sorted = P_vals[sort_idx]
    Q_sorted = Q[sort_idx]

    # Вычисление числа Рейнольдса для каждой точки
    Re = k_Re * Q_sorted / R   # Q в л/мин, R в м -> формула даёт Re

    # Отбор точек, которые по расчёту являются ламинарными (Re < 1000)
    laminar_mask = Re < 1000
    P_lam = P_sorted[laminar_mask]
    Q_lam = Q_sorted[laminar_mask]

    # Линейная регрессия для ламинарных точек (метод наименьших квадратов)
    slope, intercept = None, None
    if len(P_lam) >= 2:
        # Используем np.polyfit для линейной аппроксимации
        coeffs = np.polyfit(P_lam, Q_lam, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

    return P_sorted, Q_sorted, P_cr, P_lam, Q_lam, slope, intercept, d, l_cm

# Группировка наборов по окнам
groups = {
    'Окно 1': [1, 2, 3, 4, 11],
    'Окно 2': [5, 6, 7, 8],
    'Окно 3': [9, 10]
}

# Построение графиков
for group_name, indices in groups.items():
    n_plots = len(indices)
    if n_plots <= 2:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        axes = [axes] if n_plots == 1 else axes   # ensure iterable
    elif n_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

    fig.suptitle(group_name, fontsize=16)

    for idx, n in enumerate(indices):
        ax = axes[idx]
        P, Q, P_cr, P_lam, Q_lam, slope, intercept, d, l = process_set(n)

        # Все экспериментальные точки
        ax.scatter(P, Q, color='blue', label='Эксперимент', zorder=3)

        # Ламинарные точки (Re < 1000) выделяем зелёным
        if len(P_lam) > 0:
            ax.scatter(P_lam, Q_lam, color='green', s=50, zorder=4, label='Ламинарные (Re<1000)')

        # Линейная аппроксимация по ламинарным точкам
        if slope is not None:
            P_fit = np.linspace(0, max(P), 50)
            Q_fit = slope * P_fit + intercept
            ax.plot(P_fit, Q_fit, 'r--', label=f'Линейная аппрокс.\n slope={slope:.3f}', zorder=5)

        # Вертикальная линия теоретического критического давления
        ax.axvline(x=P_cr, color='magenta', linestyle=':', label=f'P_кр = {P_cr:.2f}', zorder=2)

        ax.set_xlabel('ΔP (усл. ед.)')
        ax.set_ylabel('Q (л/мин)')
        ax.set_title(f'd = {d} мм, l = {l} см')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Скрыть лишние подграфики
    for j in range(len(indices), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()