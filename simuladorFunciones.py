import tkinter as tk
from tkinter import messagebox, font, scrolledtext, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Importación para gráficos 3D
from matplotlib import cm # Importación para mapas de color en 3D
from scipy import stats

# --- Diccionario con descripciones y ejemplos de uso para cada distribución ---
DIST_DESCRIPTIONS = {
    "Bernoulli": "Describe un único experimento con solo dos resultados posibles: éxito (1) o fracaso (0).\n\nParámetros:\n- p: Probabilidad de éxito (0 a 1).\n\nEjemplo de Uso:\nEl resultado de lanzar una vez un dado y ver si sale un 6 (éxito, p=1/6).",
    "Binomial": "Representa el número de éxitos en 'n' ensayos independientes, donde cada ensayo solo tiene dos resultados posibles (éxito o fracaso).\n\nParámetros:\n- n: Número de ensayos (> 0).\n- p: Probabilidad de éxito (0 a 1).\n\nEjemplo de Uso:\nLanzar una moneda 10 veces (n=10) y contar cuántas veces cae 'cara' (p=0.5).",
    "Exponencial": "Modela el tiempo que transcurre entre dos eventos consecutivos en un proceso donde los eventos ocurren a una tasa constante.\n\nParámetros:\n- λ (Lambda): Tasa de ocurrencia (> 0).\n\nEjemplo de Uso:\nEl tiempo (en minutos) que esperas en la parada hasta que pasa el siguiente autobús, si llegan en promedio 2 por hora (λ=2).",
    "Normal": "La 'campana de Gauss'. Describe fenómenos naturales donde los datos se agrupan alrededor de un valor central.\n\nParámetros:\n- μ (Mu): Media o valor central.\n- σ (Sigma): Desv. Estándar (> 0).\n\nEjemplo de Uso:\nLas estaturas de los estudiantes de la FCC en la BUAP (ej. media μ=170cm, desv. estándar σ=8cm).",
    "Normal Bivariada": "Modela la relación entre dos variables que están normalmente distribuidas. Es la versión 2D de la campana de Gauss.\n\nParámetros:\n- μ_x, μ_y: Medias de X e Y.\n- σ_x, σ_y: Desv. Estándar (>0).\n- ρ (Rho): Correlación (-1 a 1).\n\nEjemplo de Uso:\nLa relación entre horas de estudio (X) y calificación (Y). Un ρ de 0.8 (ej. μ_x=10, σ_x=2, μ_y=8.5, σ_y=1) indica una fuerte relación positiva.",
    "Función Particular": "Modela una densidad de probabilidad conjunta específica definida por f(x,y) = (1/28)(2x+3y+2) en el dominio 0<x<2, 0<y<2.\n\nParámetros:\n- Ninguno. La función es fija.\n\nEjemplo de Uso:\nSimula sistemas donde la probabilidad aumenta linealmente con X e Y.",
    "Algoritmo EM (Cáncer)": "Aplica el algoritmo de Maximización de la Esperanza para encontrar los parámetros de una mezcla de dos distribuciones normales a partir de datos reales.\n\nDatos:\n- Se usará la columna 'radius (nucA)' del archivo Cancer.xlsx.\n\nIteraciones:\n- El algoritmo correrá 2 veces."
}

# --- Clase Principal de la Aplicación ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Menú Principal - Simulador de Distribuciones")
        self.root.withdraw()
        self.show_main_menu()

    def show_main_menu(self):
        self.main_menu = tk.Toplevel(self.root)
        self.main_menu.title("Menú Principal")
        self.main_menu.attributes('-fullscreen', True) # Poner en pantalla completa
        self.main_menu.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.main_menu.configure(bg="#1a1a2e") # Fondo oscuro

        title_font = font.Font(family="Helvetica", size=22, weight="bold")
        button_font = font.Font(family="Helvetica", size=14, weight="bold")
        
        content_frame = tk.Frame(self.main_menu, bg="#1a1a2e")
        content_frame.pack(expand=True)

        tk.Label(content_frame, text="Simulador de Distribuciones 📊", font=title_font, bg="#1a1a2e", fg="white").pack(pady=(30,20))

        options = {
            "Bernoulli": lambda: self.open_simulator("Bernoulli", ["Probabilidad (p)"]),
            "Binomial": lambda: self.open_simulator("Binomial", ["Ensayos (n)", "Probabilidad (p)"]),
            "Exponencial": lambda: self.open_simulator("Exponencial", ["Tasa (λ)"]),
            "Normal Univariada": lambda: self.open_simulator("Normal", ["Media (μ)", "Desv. Estándar (σ)"]),
            "Normal Bivariada (Gibbs)": lambda: self.open_simulator("Normal Bivariada", ["μ_x", "μ_y", "σ_x", "σ_y", "Correlación (ρ)"]),
            "Función Particular": lambda: self.open_simulator("Función Particular", []),
            "Algoritmo EM (Cáncer)": lambda: self.open_simulator("Algoritmo EM (Cáncer)", [])
        }
        
        columns_frame = tk.Frame(content_frame, bg="#1a1a2e")
        columns_frame.pack(pady=10)

        left_frame = tk.Frame(columns_frame, bg="#1a1a2e")
        left_frame.pack(side="left", padx=20, anchor='n')

        right_frame = tk.Frame(columns_frame, bg="#1a1a2e")
        right_frame.pack(side="right", padx=20, anchor='n')

        option_items = list(options.items())
        
        # Columna Izquierda
        for text, command in option_items[:4]:
            tk.Button(left_frame, text=text, command=command, font=button_font, bg="#9a7fdd", fg="white", width=30, height=2, relief="flat", bd=0).pack(pady=10)
        
        # Columna Derecha
        for text, command in option_items[4:]:
            tk.Button(right_frame, text=text, command=command, font=button_font, bg="#9a7fdd", fg="white", width=30, height=2, relief="flat", bd=0).pack(pady=10)

        tk.Button(content_frame, text="Salir", command=self.root.destroy, font=button_font, bg="#dc3545", fg="white", width=30, height=2, relief="flat", bd=0).pack(pady=(20,10))


    def open_simulator(self, dist_type, params):
        self.main_menu.withdraw()
        SimulatorWindow(self.root, self.main_menu, dist_type, params)

# --- Clase para la Ventana de Simulación ---
class SimulatorWindow:
    def __init__(self, root, main_menu, dist_type, params):
        self.root = root
        self.main_menu = main_menu
        self.dist_type = dist_type
        self.params = params
        self.data = None
        self.is_function_shown = False
        self.show_mixture = False
        self.show_normal1 = False
        self.show_normal2 = False
        self.em_results = None

        self.window = tk.Toplevel(root)
        self.window.title(f"Simulador - {dist_type}")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_to_menu)

        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        controls_frame = tk.Frame(main_container, bg="#2e2e5c", padx=15, pady=15)
        controls_frame.pack(side="left", fill="y", padx=(0, 20))
        
        plot_frame = tk.Frame(main_container, bg="#1a1a2e")
        plot_frame.pack(side="right", fill="both", expand=True)

        tk.Label(controls_frame, text=f"Simulador {dist_type}", font=("Helvetica", 18, "bold"), bg="#2e2e5c", fg="white").pack(pady=10)
        
        desc_label = tk.Label(controls_frame, text=DIST_DESCRIPTIONS[dist_type], justify="left", wraplength=300, bg="#2e2e5c", fg="#d1c4e9")
        desc_label.pack(pady=15, padx=5)

        self.entries = {}
        # CORRECCIÓN: Lógica para mostrar los campos de entrada necesarios
        if self.dist_type == "Algoritmo EM (Cáncer)":
            # Para EM, solo se necesita el botón de ejecutar
            tk.Button(controls_frame, text="Cargar 'Cancer.xlsx' y Ejecutar", command=self.run_em_algorithm, bg="#4CAF50", fg="white", relief="flat").pack(pady=10)
            
            em_button_frame = tk.Frame(controls_frame, bg="#2e2e5c")
            em_button_frame.pack(pady=10)
            tk.Label(em_button_frame, text="Superponer en 2D:", bg="#2e2e5c", fg="white").pack(pady=(0,5))
            
            horizontal_buttons_frame = tk.Frame(em_button_frame, bg="#2e2e5c")
            horizontal_buttons_frame.pack()

            tk.Button(horizontal_buttons_frame, text="Mezcla (Total)", command=self.toggle_mixture, bg="#9a7fdd", fg="white", relief="flat").pack(side="left", padx=3)
            tk.Button(horizontal_buttons_frame, text="Normal 1", command=self.toggle_normal1, bg="#00bcd4", fg="white", relief="flat").pack(side="left", padx=3)
            tk.Button(horizontal_buttons_frame, text="Normal 2", command=self.toggle_normal2, bg="#ff9800", fg="white", relief="flat").pack(side="left", padx=3)
        else:
            # Para todas las demás simulaciones, se necesita el tamaño de la muestra
            row = tk.Frame(controls_frame, bg="#2e2e5c")
            tk.Label(row, text="Tamaño Muestra:", width=15, anchor='w', bg="#2e2e5c", fg="white").pack(side="left")
            self.size_entry = tk.Entry(row, width=10, bg="#1a1a2e", fg="white", insertbackground="white", relief="flat")
            self.size_entry.insert(0, "1000")
            self.size_entry.pack(side="right", padx=5)
            row.pack(pady=5, fill="x")

            # Y si tienen parámetros específicos (no es el caso de Función Particular)
            if self.params:
                for param in self.params:
                    row = tk.Frame(controls_frame, bg="#2e2e5c")
                    tk.Label(row, text=f"{param}:", width=15, anchor='w', bg="#2e2e5c", fg="white").pack(side="left")
                    entry = tk.Entry(row, width=10, bg="#1a1a2e", fg="white", insertbackground="white", relief="flat")
                    entry.pack(side="right", padx=5)
                    row.pack(pady=5, fill="x")
                    self.entries[param] = entry
            
            # Botones de acción para simulaciones generales
            action_button_frame = tk.Frame(controls_frame, bg="#2e2e5c")
            action_button_frame.pack(pady=10)
            tk.Button(action_button_frame, text="Simular", command=self.run_simulation, bg="#4CAF50", fg="white", relief="flat").pack(side="left", padx=5)
            
            self.toggle_func_btn = tk.Button(action_button_frame, text="Superponer Función", command=self.toggle_theory_function, bg="#9a7fdd", fg="white", relief="flat")
            self.toggle_func_btn.pack(side="left", padx=5)

            if self.dist_type in ["Normal Bivariada", "Función Particular"]:
                self.btn_3d = tk.Button(action_button_frame, text="Visualizar en 3D", command=self.show_3d_plot, bg="#ff9800", fg="white", relief="flat")
                self.btn_3d.pack(side="left", padx=5)

        # Botones comunes
        button_frame_bottom = tk.Frame(controls_frame, bg="#2e2e5c")
        button_frame_bottom.pack(pady=10, side="bottom", fill="x")
        
        tk.Button(button_frame_bottom, text="Limpiar", command=self.clear_fields, bg="#f44336", fg="white", relief="flat").pack(fill="x", pady=5)
        tk.Button(button_frame_bottom, text="Regresar al Menú", command=self.go_to_menu, bg="#6c757d", fg="white", relief="flat").pack(fill="x", pady=5)
        
        tk.Label(controls_frame, text="Resultados:", font=("Helvetica", 12), bg="#2e2e5c", fg="white").pack(pady=(20, 5))
        self.data_text = scrolledtext.ScrolledText(controls_frame, height=10, width=40, bg="#1a1a2e", fg="white", relief="flat")
        self.data_text.pack(fill="both", expand=True)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_facecolor('#2e2e5c')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_title("El histograma aparecerá aquí")
        self.canvas.draw()
        
    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.data = None
        self.em_results = None
        self.data_text.delete('1.0', tk.END)
        self.ax.clear()
        self.ax.set_title("El histograma aparecerá aquí")
        self.canvas.draw()
        
    def go_to_menu(self):
        self.window.destroy()
        self.main_menu.deiconify()

    def run_em_algorithm(self):
        try:
            filepath = filedialog.askopenfilename(
                title="Selecciona el archivo Cancer.xlsx",
                filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
            )
            if not filepath: return

            df = pd.read_excel(filepath)
            column = 'radius (nucA)'
            if column not in df.columns:
                messagebox.showerror("Error de Columna", f"El archivo no contiene la columna '{column}'.")
                return
            
            self.data = df[column].dropna().values
            n_total = len(self.data)

            data_sorted = np.sort(self.data)
            n1 = n_total // 2
            data1, data2 = data_sorted[:n1], data_sorted[n1:]
            params = {'pi1': 0.5, 'pi2': 0.5, 'mu1': np.mean(data1), 'mu2': np.mean(data2), 'sigma1': np.std(data1), 'sigma2': np.std(data2)}
            
            results_str = "--- Iteración 0 (Valores Iniciales) ---\n"
            results_str += f"Grupo 1: μ={params['mu1']:.4f}, σ={params['sigma1']:.4f}, π={params['pi1']:.4f}\n"
            results_str += f"Grupo 2: μ={params['mu2']:.4f}, σ={params['sigma2']:.4f}, π={params['pi2']:.4f}\n\n"
            results_str += "Interpretación Inicial:\nSe asume que los datos provienen de dos grupos (Normales) de igual tamaño (π1=π2=0.5). El Grupo 1 representa los radios más pequeños y el Grupo 2 los más grandes.\n\n"

            for i in range(2):
                pdf1 = stats.norm.pdf(self.data, params['mu1'], params['sigma1'])
                pdf2 = stats.norm.pdf(self.data, params['mu2'], params['sigma2'])
                
                gamma1 = (params['pi1'] * pdf1) / (params['pi1'] * pdf1 + params['pi2'] * pdf2)
                gamma2 = 1 - gamma1

                sum_gamma1, sum_gamma2 = np.sum(gamma1), np.sum(gamma2)
                params['pi1'], params['pi2'] = sum_gamma1 / n_total, sum_gamma2 / n_total
                params['mu1'], params['mu2'] = np.sum(gamma1 * self.data) / sum_gamma1, np.sum(gamma2 * self.data) / sum_gamma2
                params['sigma1'] = np.sqrt(np.sum(gamma1 * (self.data - params['mu1'])**2) / sum_gamma1)
                params['sigma2'] = np.sqrt(np.sum(gamma2 * (self.data - params['mu2'])**2) / sum_gamma2)

                results_str += f"--- Iteración {i+1} ---\n"
                results_str += f"Grupo 1: μ={params['mu1']:.4f}, σ={params['sigma1']:.4f}, π={params['pi1']:.4f}\n"
                results_str += f"Grupo 2: μ={params['mu2']:.4f}, σ={params['sigma2']:.4f}, π={params['pi2']:.4f}\n\n"
            
            results_str += "--- Interpretación Final (Tras 2 iteraciones) ---\n"
            results_str += "El algoritmo sugiere que los datos se componen de dos grupos:\n"
            results_str += f"1. Un grupo (posiblemente benigno) que constituye el {params['pi1']:.1%} de los datos, con un radio de núcleo promedio de {params['mu1']:.2f}.\n"
            results_str += f"2. Otro grupo (posiblemente maligno) que es el {params['pi2']:.1%} restante, con un radio de núcleo promedio de {params['mu2']:.2f}.\n\n"
            results_str += "μ (mu) es la media, σ (sigma) la desviación estándar, y π (pi) es la proporción de cada grupo."
            
            self.em_results = params
            self.data_text.delete('1.0', tk.END)
            self.data_text.insert(tk.END, results_str)
            self.show_mixture, self.show_normal1, self.show_normal2 = False, False, False
            self.draw_plot()

        except Exception as e:
            messagebox.showerror("Error en Algoritmo EM", f"Ocurrió un error: {e}")

    def run_simulation(self):
        try:
            params_values = {p: float(e.get()) for p, e in self.entries.items()}
            size = int(self.size_entry.get())
            if size <= 0: raise ValueError("El tamaño de la muestra debe ser positivo.")
            
            if self.dist_type == "Bernoulli":
                p = params_values["Probabilidad (p)"]
                if not (0 <= p <= 1): raise ValueError("p debe estar entre 0 y 1.")
                self.data = np.random.binomial(1, p, size)
            elif self.dist_type == "Binomial":
                n, p = int(params_values["Ensayos (n)"]), params_values["Probabilidad (p)"]
                if not (0 <= p <= 1): raise ValueError("p debe estar entre 0 y 1.")
                self.data = np.random.binomial(n, p, size)
            elif self.dist_type == "Exponencial":
                lambda_ = params_values["Tasa (λ)"]
                if lambda_ <= 0: raise ValueError("λ debe ser positivo.")
                self.data = np.random.exponential(scale=1/lambda_, size=size)
            elif self.dist_type == "Normal":
                mu, sigma = params_values["Media (μ)"], params_values["Desv. Estándar (σ)"]
                if sigma <= 0: raise ValueError("σ debe ser positivo.")
                self.data = np.random.normal(mu, sigma, size)
            elif self.dist_type == "Función Particular":
                samples = []
                x, y = 1.0, 1.0
                burn_in = 500
                total_iter = size + burn_in

                for i in range(total_iter):
                    u1 = np.random.uniform(0, 1)
                    b_x = 3 * y + 2
                    c_x = -u1 * (4 + 3 * y)
                    x = (-b_x + np.sqrt(b_x**2 - 4 * c_x)) / 2

                    u2 = np.random.uniform(0, 1)
                    b_y = 2 * x + 2
                    c_y = -u2 * (4 * x + 10)
                    y = (-b_y + np.sqrt(b_y**2 - 6 * c_y)) / 3
                    
                    if i >= burn_in:
                        samples.append([x, y])
                self.data = np.array(samples)
            elif self.dist_type == "Normal Bivariada":
                mu_x, mu_y, sigma_x, sigma_y, rho = (params_values["μ_x"], params_values["μ_y"], params_values["σ_x"], params_values["σ_y"], params_values["Correlación (ρ)"])
                if not (-1 < rho < 1): raise ValueError("ρ debe estar entre -1 y 1.")
                if sigma_x <= 0 or sigma_y <= 0: raise ValueError("Desv. deben ser positivas.")
                samples, x, y, burn_in = np.zeros((size, 2)), 0.0, 0.0, 500
                for i in range(size + burn_in):
                    mu_cond_x = mu_x + rho * (sigma_x / sigma_y) * (y - mu_y)
                    sigma_cond_x = np.sqrt(sigma_x**2 * (1 - rho**2))
                    x = np.random.normal(mu_cond_x, sigma_cond_x)
                    mu_cond_y = mu_y + rho * (sigma_y / sigma_x) * (x - mu_x)
                    sigma_cond_y = np.sqrt(sigma_y**2 * (1 - rho**2))
                    y = np.random.normal(mu_cond_y, sigma_cond_y)
                    if i >= burn_in: samples[i - burn_in] = [x, y]
                self.data = samples
            
            self.is_function_shown = False
            self.draw_plot()
            
            self.data_text.delete('1.0', tk.END)
            if self.dist_type == "Bernoulli":
                successes = np.sum(self.data)
                total = len(self.data)
                summary_str = f"Muestra generada con éxito.\n\nDe {total} ensayos:\n- Éxitos (1): {successes}\n- Fracasos (0): {total - successes}"
                self.data_text.insert(tk.END, summary_str)
            elif self.dist_type == "Binomial":
                n = int(params_values["Ensayos (n)"])
                avg_successes = np.mean(self.data)
                summary_str = f"Muestra generada con éxito.\n\nEn {size} simulaciones de {n} ensayos cada una:\n\n- Promedio de éxitos: {avg_successes:.2f}"
                self.data_text.insert(tk.END, summary_str)
            else:
                data_str = np.array2string(self.data, precision=4, separator=', ', max_line_width=30)
                self.data_text.insert(tk.END, "Muestra generada con éxito:\n\n" + data_str)

        except (ValueError, KeyError) as e:
            messagebox.showerror("Error en los datos", f"Por favor, revisa los parámetros ingresados.\nError: {e}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
    
    def toggle_theory_function(self):
        if self.data is None:
            messagebox.showinfo("Información", "Primero debes simular los datos.")
            return
        self.is_function_shown = not self.is_function_shown
        self.draw_plot()

    def toggle_mixture(self):
        if self.data is None: messagebox.showinfo("Información", "Primero ejecuta el algoritmo."); return
        self.show_mixture = not self.show_mixture
        self.draw_plot()

    def toggle_normal1(self):
        if self.data is None: messagebox.showinfo("Información", "Primero ejecuta el algoritmo."); return
        self.show_normal1 = not self.show_normal1
        self.draw_plot()

    def toggle_normal2(self):
        if self.data is None: messagebox.showinfo("Información", "Primero ejecuta el algoritmo."); return
        self.show_normal2 = not self.show_normal2
        self.draw_plot()

    def draw_plot(self):
        if self.data is None: return
        self.ax.clear()
        params = {p: float(e.get()) for p, e in self.entries.items()} if self.entries else {}

        if self.dist_type == "Bernoulli":
            p = params["Probabilidad (p)"]
            bins = np.arange(-0.5, 2.5, 1)
            self.ax.hist(self.data, bins=bins, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown:
                self.ax.plot([0, 1], [1-p, p], 'yo', markersize=10, linestyle='--', label="PMF Teórica")
            self.ax.set_xticks([0, 1])
            self.ax.set_title(f"Distribución Bernoulli (p={p:.2f})", color="white")
        
        elif self.dist_type == "Binomial":
            n, p = int(params["Ensayos (n)"]), params["Probabilidad (p)"]
            bins = np.arange(-0.5, n + 1.5, 1)
            self.ax.hist(self.data, bins=bins, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown:
                x = np.arange(0, n + 1)
                pmf = stats.binom.pmf(x, n, p)
                self.ax.plot(x, pmf, 'yo-', label="PMF Teórica")
            self.ax.set_title(f"Distribución Binomial (n={n}, p={p:.2f})", color="white")
        
        elif self.dist_type == "Exponencial":
            lambda_ = params["Tasa (λ)"]
            self.ax.hist(self.data, bins=30, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown:
                x = np.linspace(self.data.min(), self.data.max(), 100)
                pdf = stats.expon.pdf(x, scale=1/lambda_)
                self.ax.plot(x, pdf, 'y-', lw=2, label="PDF Teórica")
            self.ax.set_title(f"Distribución Exponencial (λ={lambda_:.2f})", color="white")

        elif self.dist_type == "Normal":
            mu, sigma = params["Media (μ)"], params["Desv. Estándar (σ)"]
            self.ax.hist(self.data, bins=30, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown:
                x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
                pdf = stats.norm.pdf(x, mu, sigma)
                self.ax.plot(x, pdf, 'y-', lw=2, label="PDF Teórica")
            self.ax.set_title(f"Distribución Normal (μ={mu:.2f}, σ={sigma:.2f})", color="white")

        elif self.dist_type == "Función Particular":
            self.ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.5, s=15, color="#9a7fdd", edgecolor='none')
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            if self.is_function_shown:
                x_grid = np.linspace(0, 2, 100)
                y_grid = np.linspace(0, 2, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = (2*X + 3*Y + 2) / 28
                self.ax.contour(X, Y, Z, colors='yellow', alpha=0.7)
            self.ax.set_title("Función Particular f(x,y)", color="white")

        elif self.dist_type == "Normal Bivariada":
            mu_x, mu_y, sigma_x, sigma_y, rho = (params["μ_x"], params["μ_y"], params["σ_x"], params["σ_y"], params["Correlación (ρ)"])
            self.ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.5, s=15, color="#9a7fdd", edgecolor='none')
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            if self.is_function_shown:
                x_grid = np.linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 100)
                y_grid = np.linspace(mu_y - 3*sigma_y, mu_y + 3*sigma_y, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                pos = np.dstack((X, Y))
                cov = [[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]
                rv = stats.multivariate_normal([mu_x, mu_y], cov)
                self.ax.contour(X, Y, rv.pdf(pos), colors='yellow', alpha=0.7)
            self.ax.set_title("Normal Bivariada (Muestreo de Gibbs)", color="white")

        elif self.dist_type == "Algoritmo EM (Cáncer)" and self.em_results:
            self.ax.hist(self.data, bins=50, density=True, alpha=0.6, label="Datos 'radius (nucA)'", color="#9a7fdd")
            x = np.linspace(self.data.min(), self.data.max(), 500)
            p = self.em_results
            
            pdf1 = stats.norm.pdf(x, p['mu1'], p['sigma1']) * p['pi1']
            pdf2 = stats.norm.pdf(x, p['mu2'], p['sigma2']) * p['pi2']
            pdf_total = pdf1 + pdf2

            if self.show_mixture:
                self.ax.plot(x, pdf_total, 'y-', lw=3, label="Mezcla de Normales (Total)")
            if self.show_normal1:
                self.ax.plot(x, pdf1, 'c--', lw=2, label=f"Normal 1 (μ={p['mu1']:.2f})")
            if self.show_normal2:
                self.ax.plot(x, pdf2, 'm--', lw=2, label=f"Normal 2 (μ={p['mu2']:.2f})")
            
            self.ax.set_title("Resultado del Algoritmo EM", color="white")
            if self.show_mixture or self.show_normal1 or self.show_normal2:
                self.ax.legend()


        if self.dist_type not in ["Normal Bivariada", "Función Particular", "Algoritmo EM (Cáncer)"]:
            if self.is_function_shown: self.ax.legend()
            self.ax.set_xlabel("Valor")
            self.ax.set_ylabel("Densidad / Probabilidad")
        
        self.canvas.draw()

    def show_3d_plot(self):
        if self.data is None:
            messagebox.showinfo("Información", "Primero debes simular los datos.")
            return
        try:
            win3d = tk.Toplevel(self.window)
            win3d.title(f"Visualización 3D - {self.dist_type}")
            win3d.geometry("800x700")

            fig3d = plt.figure(figsize=(8, 7))
            fig3d.patch.set_facecolor('#1a1a2e')
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.set_facecolor('#1a1a2e')

            title = ""
            if self.dist_type == "Normal Bivariada":
                params = {p: float(e.get()) for p, e in self.entries.items()}
                mu_x, mu_y, sigma_x, sigma_y, rho = (params["μ_x"], params["μ_y"], params["σ_x"], params["σ_y"], params["Correlación (ρ)"])
                
                x_range = [mu_x - 3.5*sigma_x, mu_x + 3.5*sigma_x]
                y_range = [mu_y - 3.5*sigma_y, mu_y + 3.5*sigma_y]
                x_grid = np.linspace(x_range[0], x_range[1], 70)
                y_grid = np.linspace(y_range[0], y_range[1], 70)
                X, Y = np.meshgrid(x_grid, y_grid)
                pos = np.dstack((X, Y))
                cov = [[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]
                rv = stats.multivariate_normal([mu_x, mu_y], cov)
                Z = rv.pdf(pos)
                ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
                
                hist, xedges, yedges = np.histogram2d(self.data[:, 0], self.data[:, 1], bins=20, range=[x_range, y_range], density=True)
                xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1]-xedges[0])/2., yedges[:-1] + (yedges[1]-yedges[0])/2., indexing="ij")
                xpos = xpos.ravel()
                ypos = ypos.ravel()
                zpos = 0
                dx = (xedges[1]-xedges[0]) * np.ones_like(zpos)
                dy = (yedges[1]-yedges[0]) * np.ones_like(zpos)
                dz = hist.ravel()
                ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='#9a7fdd', alpha=0.7)
                
                title = "Superficie (Teórica) y Muestra (Histograma)"
            
            elif self.dist_type == "Función Particular":
                x_grid = np.linspace(0, 2, 70)
                y_grid = np.linspace(0, 2, 70)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = (2*X + 3*Y + 2) / 28
                ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')

                ax3d.scatter(self.data[:, 0], self.data[:, 1], 0, c='#9a7fdd', alpha=0.4, s=10)
                
                title = "Superficie (Teórica) y Muestra (Puntos)"

            ax3d.set_xlabel('Eje X'); ax3d.xaxis.label.set_color('white')
            ax3d.set_ylabel('Eje Y'); ax3d.yaxis.label.set_color('white')
            ax3d.set_zlabel('Densidad'); ax3d.zaxis.label.set_color('white')
            ax3d.set_title(title, color="white")
            
            ax3d.tick_params(axis='x', colors='white')
            ax3d.tick_params(axis='y', colors='white')
            ax3d.tick_params(axis='z', colors='white')
            
            canvas3d = FigureCanvasTkAgg(fig3d, master=win3d)
            canvas3d.get_tk_widget().pack(fill="both", expand=True)
            canvas3d.draw()

        except (ValueError, KeyError) as e:
            messagebox.showerror("Error en los datos", f"No se pueden generar los datos para la gráfica 3D.\nError: {e}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error al crear la gráfica 3D: {e}")

# --- Ejecución de la aplicación ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = App(main_root)
    main_root.mainloop()

