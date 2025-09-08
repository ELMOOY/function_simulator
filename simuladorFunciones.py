import tkinter as tk
from tkinter import messagebox, font, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Importación para gráficos 3D
from matplotlib import cm # Importación para mapas de color en 3D
from scipy import stats

# --- Diccionario con descripciones y ejemplos de uso para cada distribución ---
DIST_DESCRIPTIONS = {
    "Puntual": "Describe un evento sin aleatoriedad, donde el resultado es siempre el mismo.\n\nParámetros:\n- k: El valor constante.\n\nEjemplo:\nEl número de soles en nuestro sistema solar. El valor siempre es 1 (k=1).",
    "Binomial": "Representa el número de éxitos en 'n' ensayos independientes, donde cada ensayo solo tiene dos resultados posibles (éxito o fracaso).\n\nParámetros:\n- n: Número de ensayos (> 0).\n- p: Probabilidad de éxito (0 a 1).\n\nEjemplo:\nLanzar una moneda 10 veces (n=10) y contar cuántas veces cae 'cara' (p=0.5).",
    "Exponencial": "Modela el tiempo que transcurre entre dos eventos consecutivos en un proceso donde los eventos ocurren a una tasa constante.\n\nParámetros:\n- λ (Lambda): Tasa de ocurrencia (> 0).\n\nEjemplo:\nEl tiempo (en minutos) que esperas en la parada hasta que pasa el siguiente lobobus, si llegan en promedio 2 por hora (λ=2).",
    "Normal": "La 'campana de Gauss'. Describe fenómenos naturales donde los datos se agrupan alrededor de un valor central.\n\nParámetros:\n- μ (Mu): Media o valor central.\n- σ (Sigma): Desv. Estándar (> 0).\n\nEjemplo:\nLas estaturas de los estudiantes (ej. media μ=170cm, desv. estándar σ=8cm).",
    "Normal Bivariada": "Modela la relación entre dos variables que están normalmente distribuidas. Es la versión 2D de la campana de Gauss.\n\nParámetros:\n- μ_x, μ_y: Medias de X e Y.\n- σ_x, σ_y: Desv. Estándar (>0).\n- ρ (Rho): Correlación (-1 a 1).\n\nEjemplo:\nLa relación entre horas de estudio (X) y calificación (Y). Un ρ de 0.8 (ej. μ_x=10, σ_x=2, μ_y=8.5, σ_y=1) indica una fuerte relación positiva: a más horas de estudio, tiende a haber una calificación más alta.",
    "Beta": "Modela valores que representan una proporción o un porcentaje, por lo que siempre están entre 0 y 1.\n\nParámetros:\n- α (Alfa): Parámetro de forma (>0).\n- β (Beta): Parámetro de forma (>0).\n\nEjemplo:\nEl porcentaje de bateo de un. Si tiene más éxitos que fallos, la curva se carga a la derecha (ej. α=8, β=2)."
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
        
        # Frame para centrar el contenido
        content_frame = tk.Frame(self.main_menu, bg="#1a1a2e")
        content_frame.pack(expand=True)

        tk.Label(content_frame, text="Simulador de Distribuciones", font=title_font, bg="#1a1a2e", fg="white").pack(pady=(30,20))

        options = {
            "Puntual": lambda: self.open_simulator("Puntual", ["Valor (k)"]),
            "Binomial": lambda: self.open_simulator("Binomial", ["Ensayos (n)", "Probabilidad (p)"]),
            "Exponencial": lambda: self.open_simulator("Exponencial", ["Tasa (λ)"]),
            "Normal Univariada": lambda: self.open_simulator("Normal", ["Media (μ)", "Desv. Estándar (σ)"]),
            "Normal Bivariada (Gibbs)": lambda: self.open_simulator("Normal Bivariada", ["μ_x", "μ_y", "σ_x", "σ_y", "Correlación (ρ)"]),
            "Beta": lambda: self.open_simulator("Beta", ["Alfa (α)", "Beta (β)"])
        }
        
        # --- Frame para las dos columnas de botones ---
        columns_frame = tk.Frame(content_frame, bg="#1a1a2e")
        columns_frame.pack(pady=10)

        left_frame = tk.Frame(columns_frame, bg="#1a1a2e")
        left_frame.pack(side="left", padx=20, anchor='n')

        right_frame = tk.Frame(columns_frame, bg="#1a1a2e")
        right_frame.pack(side="right", padx=20, anchor='n')

        option_items = list(options.items())

        # Columna Izquierda (primeras 3 opciones)
        for text, command in option_items[:3]:
            tk.Button(left_frame, text=text, command=command, font=button_font, bg="#9a7fdd", fg="white", width=30, height=2, relief="flat", bd=0).pack(pady=10)
        
        # Columna Derecha (siguientes 3 opciones)
        for text, command in option_items[3:]:
            tk.Button(right_frame, text=text, command=command, font=button_font, bg="#9a7fdd", fg="white", width=30, height=2, relief="flat", bd=0).pack(pady=10)

        # Botón para salir de la aplicación, centrado debajo
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
        self.data = None # Para guardar los datos simulados
        self.is_function_shown = False # Controla si la función teórica está visible

        self.window = tk.Toplevel(root)
        self.window.title(f"Simulador - {dist_type}")
        self.window.attributes('-fullscreen', True) # Poner en pantalla completa
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_to_menu)

        # --- Frames para organizar la interfaz ---
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
        for param in self.params:
            row = tk.Frame(controls_frame, bg="#2e2e5c")
            tk.Label(row, text=f"{param}:", width=15, anchor='w', bg="#2e2e5c", fg="white").pack(side="left")
            entry = tk.Entry(row, width=10, bg="#1a1a2e", fg="white", insertbackground="white", relief="flat")
            entry.pack(side="right", padx=5)
            row.pack(pady=5, fill="x")
            self.entries[param] = entry

        row = tk.Frame(controls_frame, bg="#2e2e5c")
        tk.Label(row, text="Tamaño Muestra:", width=15, anchor='w', bg="#2e2e5c", fg="white").pack(side="left")
        self.size_entry = tk.Entry(row, width=10, bg="#1a1a2e", fg="white", insertbackground="white", relief="flat")
        self.size_entry.insert(0, "1000")
        self.size_entry.pack(side="right", padx=5)
        row.pack(pady=5, fill="x")
        
        button_frame = tk.Frame(controls_frame, bg="#2e2e5c")
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Simular", command=self.run_simulation, bg="#4CAF50", fg="white", relief="flat").pack(side="left", padx=5)
        
        if self.dist_type != "Puntual":
            self.toggle_func_btn = tk.Button(button_frame, text="Superponer Función", command=self.toggle_theory_function, bg="#9a7fdd", fg="white", relief="flat")
            self.toggle_func_btn.pack(side="left", padx=5)

        if self.dist_type == "Normal Bivariada":
            self.btn_3d = tk.Button(button_frame, text="Visualizar en 3D", command=self.show_3d_plot, bg="#ff9800", fg="white", relief="flat")
            self.btn_3d.pack(side="left", padx=5)

        tk.Button(controls_frame, text="Limpiar", command=self.clear_fields, bg="#f44336", fg="white", relief="flat").pack(fill="x", pady=5)
        tk.Button(controls_frame, text="Regresar al Menú", command=self.go_to_menu, bg="#6c757d", fg="white", relief="flat").pack(fill="x", pady=20, side="bottom")

        tk.Label(controls_frame, text="Muestra Generada:", font=("Helvetica", 12), bg="#2e2e5c", fg="white").pack(pady=(20, 5))
        self.data_text = scrolledtext.ScrolledText(controls_frame, height=10, width=30, bg="#1a1a2e", fg="white", relief="flat")
        self.data_text.pack(fill="both", expand=True)

        # --- Lienzo para la gráfica de Matplotlib ---
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
        self.data_text.delete('1.0', tk.END)
        self.ax.clear()
        self.ax.set_title("El histograma aparecerá aquí")
        self.canvas.draw()
        
    def go_to_menu(self):
        self.window.destroy()
        self.main_menu.deiconify()

    def run_simulation(self):
        """Genera los datos y dibuja el histograma base."""
        try:
            params_values = {p: float(e.get()) for p, e in self.entries.items()}
            size = int(self.size_entry.get())
            if size <= 0: raise ValueError("El tamaño de la muestra debe ser positivo.")
            
            if self.dist_type == "Puntual":
                self.data = np.full(size, params_values["Valor (k)"])
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
            elif self.dist_type == "Beta":
                a, b = params_values["Alfa (α)"], params_values["Beta (β)"]
                if a <= 0 or b <= 0: raise ValueError("α y β deben ser positivos.")
                self.data = np.random.beta(a, b, size)
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

    def draw_plot(self):
        if self.data is None: return
        self.ax.clear()
        params = {p: float(e.get()) for p, e in self.entries.items()}

        if self.dist_type == "Puntual":
            k = params["Valor (k)"]
            self.ax.hist(self.data, bins=np.arange(k-0.5, k+1.5, 1), density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            self.ax.set_title(f"Distribución Puntual (k={k})", color="white")
        
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

        elif self.dist_type == "Beta":
            a, b = params["Alfa (α)"], params["Beta (β)"]
            self.ax.hist(self.data, bins=30, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown:
                x = np.linspace(0, 1, 100)
                pdf = stats.beta.pdf(x, a, b)
                self.ax.plot(x, pdf, 'y-', lw=2, label="PDF Teórica")
            self.ax.set_title(f"Distribución Beta (α={a:.2f}, β={b:.2f})", color="white")

        elif self.dist_type == "Normal Bivariada":
            mu_x, mu_y, sigma_x, sigma_y, rho = (params["μ_x"], params["μ_y"], params["σ_x"], params["σ_y"], params["Correlación (ρ)"])
            self.ax.hist2d(self.data[:, 0], self.data[:, 1], bins=40, cmap='Purples')
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

        if self.dist_type != "Normal Bivariada":
            if self.is_function_shown: self.ax.legend()
            self.ax.set_xlabel("Valor")
            self.ax.set_ylabel("Densidad / Probabilidad")
        
        self.canvas.draw()

    def show_3d_plot(self):
        if self.data is None:
            messagebox.showinfo("Información", "Primero debes simular los datos.")
            return
        try:
            params = {p: float(e.get()) for p, e in self.entries.items()}
            mu_x, mu_y, sigma_x, sigma_y, rho = (params["μ_x"], params["μ_y"], params["σ_x"], params["σ_y"], params["Correlación (ρ)"])

            win3d = tk.Toplevel(self.window)
            win3d.title("Visualización 3D - Normal Bivariada")
            win3d.geometry("800x700")

            fig3d = plt.figure(figsize=(8, 7))
            fig3d.patch.set_facecolor('#1a1a2e')
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.set_facecolor('#1a1a2e')

            x_grid = np.linspace(mu_x - 3.5*sigma_x, mu_x + 3.5*sigma_x, 70)
            y_grid = np.linspace(mu_y - 3.5*sigma_y, mu_y + 3.5*sigma_y, 70)
            X, Y = np.meshgrid(x_grid, y_grid)
            pos = np.dstack((X, Y))
            cov = [[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]
            rv = stats.multivariate_normal([mu_x, mu_y], cov)
            Z = rv.pdf(pos)
            
            ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, edgecolor='none')

            ax3d.set_xlabel('Eje X'); ax3d.xaxis.label.set_color('white')
            ax3d.set_ylabel('Eje Y'); ax3d.yaxis.label.set_color('white')
            ax3d.set_zlabel('Densidad'); ax3d.zaxis.label.set_color('white')
            ax3d.set_title("Superficie de Densidad Normal Bivariada", color="white")
            
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

