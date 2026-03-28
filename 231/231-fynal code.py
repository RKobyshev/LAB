import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

# ------------------------------------------------------------
# Данные эксперимента (давление в 10^-4 торр)
pressure_data = [
    0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97,
    0.97, 0.97, 0.965, 0.96, 0.98, 0.97, 0.99, 1.0, 1.1, 1.1,
    1.2, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.6, 1.7, 1.7,
    1.8, 1.9, 1.9, 2.0, 2.1, 2.1, 2.2, 2.3, 2.3, 2.4,
    2.45, 2.5, 2.5, 2.6, 2.7, 2.71, 2.8, 2.9, 2.9, 2.9,
    3.0, 3.1, 3.1, 3.2, 3.2, 3.3, 3.4, 3.4, 3.5, 3.5,
    3.6, 3.7, 3.7, 3.8, 3.9, 3.9, 3.9, 4.0, 4.1, 4.1,
    4.2, 4.2, 4.3, 4.3, 4.4, 4.5, 4.5, 4.6, 4.6, 4.7,
    4.7, 4.8, 4.85, 4.9, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2,
    5.3, 5.4, 5.4, 5.5, 5.5, 5.6, 5.6, 5.7, 5.7, 5.8,
    5.9, 5.9, 6.0, 6.0, 6.1, 6.1, 6.1, 5.8, 5.3, 4.6,
    3.9, 3.0, 2.7, 2.4, 2.1, 1.9, 1.7, 1.5, 1.45, 1.4,
    1.25, 1.1, 1.15, 1.2, 1.3, 1.1, 1.1, 1.1, 1.1, 1.1,
    1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0
]

# ------------------------------------------------------------
# Параметры установки (можно менять)
V = 1.16                     # объём высоковакуумной части, л
L_cap = 10.8                 # длина капилляра, см
d_cap = 0.8                  # диаметр капилляра, мм
r_cap = (d_cap / 10) / 2     # радиус в см: 0.8 мм = 0.08 см -> 0.04 см
P_fv = 1.6e-3                # давление в форвакуумной части, торр

# Теоретическая проводимость капилляра (Кнудсен)
C_cap_theor = 12.1 * (r_cap**3) / L_cap   # л/с

# Вспомогательные массивы
t = np.arange(len(pressure_data))
P_torr = np.array(pressure_data) * 1e-4   # торр

# Индекс максимума (начало спада) – нужен для ограничения ползунков
idx_max = np.argmax(P_torr)

# Предельное давление (среднее по первым 10 точкам до роста)
P_pr = np.mean(P_torr[:10])

# ------------------------------------------------------------
class VacuumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ вакуумных данных")
        self.root.geometry("1100x800")

        # Начальные значения для ползунков
        self.start_growth = 10          # исключаем начальное плато
        self.end_growth = idx_max
        self.start_decay = idx_max
        self.end_decay = len(t)-1

        # График предварительного просмотра
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Панель управления
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Ползунки для участка роста (натекание)
        ttk.Label(control_frame, text="Рост (натекание) – начало (с):").grid(row=0, column=0, sticky=tk.W)
        self.slider_g_start = tk.Scale(control_frame, from_=0, to=idx_max, orient=tk.HORIZONTAL,
                                       length=400, command=self.update_preview)
        self.slider_g_start.set(self.start_growth)
        self.slider_g_start.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Рост (натекание) – конец (с):").grid(row=1, column=0, sticky=tk.W)
        self.slider_g_end = tk.Scale(control_frame, from_=0, to=idx_max, orient=tk.HORIZONTAL,
                                     length=400, command=self.update_preview)
        self.slider_g_end.set(self.end_growth)
        self.slider_g_end.grid(row=1, column=1, padx=5)

        # Ползунки для участка спада (откачка)
        ttk.Label(control_frame, text="Спад (откачка) – начало (с):").grid(row=2, column=0, sticky=tk.W)
        self.slider_d_start = tk.Scale(control_frame, from_=idx_max, to=len(t)-1, orient=tk.HORIZONTAL,
                                       length=400, command=self.update_preview)
        self.slider_d_start.set(self.start_decay)
        self.slider_d_start.grid(row=2, column=1, padx=5)

        ttk.Label(control_frame, text="Спад (откачка) – конец (с):").grid(row=3, column=0, sticky=tk.W)
        self.slider_d_end = tk.Scale(control_frame, from_=idx_max+1, to=len(t)-1, orient=tk.HORIZONTAL,
                                     length=400, command=self.update_preview)
        self.slider_d_end.set(self.end_decay)
        self.slider_d_end.grid(row=3, column=1, padx=5)

        # Кнопка вывода данных
        self.finish_btn = ttk.Button(control_frame, text="Вывести данные", command=self.finish_and_plot)
        self.finish_btn.grid(row=4, column=0, columnspan=2, pady=10)

        # Информационная метка
        self.info_text = tk.StringVar()
        info_label = ttk.Label(control_frame, textvariable=self.info_text, justify=tk.LEFT)
        info_label.grid(row=5, column=0, columnspan=2, pady=5)

        self.update_preview()

    def update_preview(self, *args):
        """Обновляет предварительный график с вертикальными линиями для всех границ"""
        try:
            sg = int(self.slider_g_start.get())
            eg = int(self.slider_g_end.get())
            sd = int(self.slider_d_start.get())
            ed = int(self.slider_d_end.get())
        except:
            return

        if sg >= eg or sd >= ed:
            self.info_text.set("Ошибка: начало должно быть меньше конца.")
            return

        self.start_growth, self.end_growth = sg, eg
        self.start_decay, self.end_decay = sd, ed

        self.ax.clear()
        self.ax.plot(t, P_torr*1e4, 'b-', linewidth=1, label='Давление')

        # Линии для роста
        self.ax.axvline(x=sg, color='orange', linestyle='--', alpha=0.7, label='Начало роста')
        self.ax.axvline(x=eg, color='orange', linestyle='--', alpha=0.7, label='Конец роста')

        # Линии для спада
        self.ax.axvline(x=sd, color='red', linestyle='--', alpha=0.7, label='Начало спада')
        self.ax.axvline(x=ed, color='red', linestyle='--', alpha=0.7, label='Конец спада')

        # Вертикальная линия максимума
        self.ax.axvline(x=idx_max, color='green', linestyle=':', alpha=0.5, label='Максимум')

        self.ax.set_xlabel('Время, с')
        self.ax.set_ylabel('Давление, ×10⁻⁴ торр')
        self.ax.set_title('Полный график давления (выделены интервалы)')
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

        self.info_text.set(f"Рост: {sg}–{eg} с, Спад: {sd}–{ed} с")

    def finish_and_plot(self):
        """Закрыть GUI, построить финальные графики и вывести результаты в консоль"""
        sg = self.start_growth
        eg = self.end_growth
        sd = self.start_decay
        ed = self.end_decay

        self.root.quit()
        self.root.destroy()

        # ------------------- 1. Полный график -------------------
        plt.figure(figsize=(10, 5))
        plt.plot(t, P_torr*1e4, 'b-', linewidth=1, label='Давление')
        plt.axvline(x=sg, color='orange', linestyle='--', label='Начало роста')
        plt.axvline(x=eg, color='orange', linestyle='--', label='Конец роста')
        plt.axvline(x=sd, color='red', linestyle='--', label='Начало спада')
        plt.axvline(x=ed, color='red', linestyle='--', label='Конец спада')
        plt.axvline(x=idx_max, color='green', linestyle=':', alpha=0.5, label='Максимум')
        plt.xlabel('Время, с')
        plt.ylabel('Давление, ×10⁻⁴ торр')
        plt.title('Полный график давления с выбранными интервалами')
        plt.grid(True)
        plt.legend()
        plt.show()

        # ------------------- 2. График роста (линейная аппроксимация) -------------------
        t_grow = t[sg:eg+1]
        P_grow = P_torr[sg:eg+1]
        if len(t_grow) >= 2:
            slope_g, intercept_g, r_g, _, _ = stats.linregress(t_grow, P_grow)
            Q_sum = V * slope_g
            plt.figure(figsize=(8, 5))
            plt.plot(t_grow, P_grow*1e4, 'bo', markersize=3, label='Данные')
            plt.plot(t_grow, (intercept_g + slope_g*t_grow)*1e4, 'r-', linewidth=2,
                     label=f'Аппроксимация: dP/dt = {slope_g:.3e} торр/с, R² = {r_g**2:.4f}')
            plt.xlabel('Время, с')
            plt.ylabel('Давление, ×10⁻⁴ торр')
            plt.title('Участок роста (натекание)')
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            slope_g = np.nan
            Q_sum = np.nan
            r_g = np.nan
            print("Недостаточно точек для аппроксимации роста.")

        # ------------------- 3. График спада (полулогарифмический) -------------------
        t_decay = t[sd:ed+1]
        P_decay = P_torr[sd:ed+1]
        lnP_decay = np.log(P_decay)
        valid = np.isfinite(lnP_decay)
        t_valid = t_decay[valid]
        lnP_valid = lnP_decay[valid]

        if len(t_valid) >= 2:
            slope_d, intercept_d, r_d, _, _ = stats.linregress(t_valid, lnP_valid)
            tau = -1 / slope_d
            W_exp = V / tau
            plt.figure(figsize=(8, 5))
            plt.plot(t_valid, lnP_valid, 'bo', markersize=3, label='Данные')
            plt.plot(t_valid, intercept_d + slope_d*t_valid, 'r-', linewidth=2,
                     label=f'Аппроксимация: τ = {tau:.2f} с, R² = {r_d**2:.4f}')
            plt.xlabel('Время, с')
            plt.ylabel('ln(P) (P в торр)')
            plt.title('Участок спада (откачка)')
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            tau = np.nan
            W_exp = np.nan
            r_d = np.nan
            print("Недостаточно точек для аппроксимации спада.")

        # ------------------- Определение P_уст (плато после спада) -------------------
        # Все точки с индексом > ed считаем плато (стационарное давление при открытом капилляре и работающем насосе)
        plateau_indices = np.where(t > ed)[0]
        if len(plateau_indices) > 0:
            P_ust = np.mean(P_torr[plateau_indices])   # торр
        else:
            P_ust = np.nan

        # ------------------- Расчёт W_иск (метод искусственной течи) -------------------
        if not np.isnan(P_ust) and not np.isnan(C_cap_theor) and not np.isnan(P_fv) and not np.isnan(P_pr):
            Q_cap = C_cap_theor * (P_fv - P_ust)
            denom = P_ust - P_pr
            if denom > 0:
                W_art = Q_cap / denom
            else:
                W_art = np.nan
        else:
            W_art = np.nan
            Q_cap = np.nan

        # ------------------- Вывод результатов в консоль -------------------
        print("\n" + "="*65)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ (с обновлёнными параметрами установки)")
        print("="*65)
        print(f"Объём высоковакуумной части V = {V:.3f} л")
        print(f"Давление в форвакуумной части P_фв = {P_fv:.2e} торр")
        print(f"Размеры капилляра: L = {L_cap:.1f} см, d = {d_cap:.1f} мм")
        print(f"Теоретическая проводимость капилляра C_кап = {C_cap_theor:.3e} л/с")

        print("\n--- Участок роста (натекание) ---")
        if not np.isnan(slope_g):
            print(f"  Интервал: {sg} – {eg} с")
            print(f"  dP/dt = {slope_g:.3e} торр/с")
            print(f"  Суммарный поток Q_сум = {Q_sum:.3e} торр·л/с")
            print(f"  Коэффициент детерминации R² = {r_g**2:.4f}")
        else:
            print("  Аппроксимация не удалась.")

        print("\n--- Участок спада (откачка) ---")
        if not np.isnan(tau):
            print(f"  Интервал: {sd} – {ed} с")
            print(f"  Постоянная времени τ = {tau:.2f} с")
            print(f"  Эффективная скорость откачки (экспоненциальный спад) W_эксп = {W_exp:.3f} л/с")
            print(f"  Коэффициент детерминации R² = {r_d**2:.4f}")
        else:
            print("  Аппроксимация не удалась.")

        print("\n--- Метод искусственной течи ---")
        print(f"  Предельное давление P_пр = {P_pr:.2e} торр")
        print(f"  Установившееся давление (плато после спада) P_уст = {P_ust:.2e} торр")
        if not np.isnan(Q_cap):
            print(f"  Поток через капилляр Q_кап = C_кап·(P_фв - P_уст) = {Q_cap:.3e} торр·л/с")
        else:
            print("  Невозможно рассчитать поток через капилляр.")
        if not np.isnan(W_art):
            print(f"  Скорость откачки W_иск = {W_art:.3f} л/с")
        else:
            print("  Скорость откачки по методу искусственной течи не определена (знаменатель ≤ 0 или нет данных).")

        print("\n--- Сравнение ---")
        if not np.isnan(W_exp) and not np.isnan(W_art):
            print(f"  W_эксп = {W_exp:.3f} л/с")
            print(f"  W_иск  = {W_art:.3f} л/с")
            print(f"  Относительное расхождение: {abs(W_exp - W_art)/W_exp*100:.1f} %")
        print("="*65)

if __name__ == "__main__":
    root = tk.Tk()
    app = VacuumApp(root)
    root.mainloop()