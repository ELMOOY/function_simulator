import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import stats
from scipy.stats import multivariate_normal
from tkinter import messagebox, font, scrolledtext, filedialog, OptionMenu, ttk
from PIL import Image, ImageTk

# --- Constante para la Galería ---
GALLERY_DIR = "em_gallery"
CARPETA_DIAGNOSTICO = "diagnostico"

# --- Diccionario con descripciones ---
DIST_DESCRIPTIONS = {
    "Bernoulli": "Describe un único experimento con solo dos resultados posibles: éxito (1) o fracaso (0).\n\nParámetros:\n- p: Probabilidad de éxito (0 a 1).",
    "Binomial": "Representa el número de éxitos en 'n' ensayos independientes.\n\nParámetros:\n- n: Número de ensayos (> 0).\n- p: Probabilidad de éxito (0 a 1).",
    "Exponencial": "Modela el tiempo entre dos eventos consecutivos a una tasa constante.\n\nParámetros:\n- λ (Lambda): Tasa de ocurrencia (> 0).",
    "Normal": "La 'campana de Gauss'.\n\nParámetros:\n- μ (Mu): Media o valor central.\n- σ (Sigma): Desv. Estándar (> 0).",
    "Normal Bivariada": "Versión 2D de la campana de Gauss.\n\nParámetros:\n- μ_x, μ_y: Medias.\n- σ_x, σ_y: Desv. Estándar.\n- ρ (Rho): Correlación (-1 a 1).",
    "Función Particular": "Densidad conjunta específica f(x,y) = (1/28)(2x+3y+2) en 0<x<2, 0<y<2.",
    "Algoritmo EM (Cáncer)": "Encuentra parámetros de una mezcla de dos normales (ej. Benigno/Maligno) a partir de datos."
}

# --- Clase Principal de la Aplicación ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Menú Principal - Simulador")
        try:
            os.makedirs(GALLERY_DIR, exist_ok=True)
            os.makedirs(CARPETA_DIAGNOSTICO, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"No se pudo crear carpeta '{GALLERY_DIR}':\n{e}")
            messagebox.showerror("Error", f"No se pudo crear carpeta '{CARPETA_DIAGNOSTICO}':\n{e}")

        

        self.root.withdraw()
        self.show_main_menu()

    

    def show_main_menu(self):
        self.main_menu = tk.Toplevel(self.root)
        self.main_menu.title("Menú Principal")
        self.main_menu.attributes('-fullscreen', True)
        self.main_menu.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.main_menu.configure(bg="#1a1a2e")

        title_font = font.Font(family="Helvetica", size=22, weight="bold")
        button_font = font.Font(family="Helvetica", size=14, weight="bold")

        content_frame = tk.Frame(self.main_menu, bg="#1a1a2e")
        content_frame.pack(expand=True)

        tk.Label(content_frame, text="Simulador de Distribuciones & Clasificador", font=title_font, bg="#1a1a2e", fg="white").pack(pady=(30, 20))

        simulator_commands = {
            "Bernoulli": lambda: self.open_simulator("Bernoulli", ["Probabilidad (p)"]),
            "Binomial": lambda: self.open_simulator("Binomial", ["Ensayos (n)", "Probabilidad (p)"]),
            "Exponencial": lambda: self.open_simulator("Exponencial", ["Tasa (λ)"]),
            "Normal Univariada": lambda: self.open_simulator("Normal", ["Media (μ)", "Desv. Estándar (σ)"]),
            "Normal Bivariada (Gibbs)": lambda: self.open_simulator("Normal Bivariada", ["μ_x", "μ_y", "σ_x", "σ_y", "Correlación (ρ)"]),
            "Función Particular": lambda: self.open_simulator("Función Particular", []),
            "Algoritmo EM (Cáncer)": lambda: self.open_simulator("Algoritmo EM (Cáncer)", ["Número de Iteraciones"])
        }

        columns_frame = tk.Frame(content_frame, bg="#1a1a2e")
        columns_frame.pack(pady=10)

        left_frame = tk.Frame(columns_frame, bg="#1a1a2e")
        left_frame.pack(side="left", padx=20, anchor='n')
        right_frame = tk.Frame(columns_frame, bg="#1a1a2e")
        right_frame.pack(side="right", padx=20, anchor='n')

        simulator_items = list(simulator_commands.items())

        for text, command in simulator_items[:4]:
            tk.Button(left_frame, text=text, command=command, font=button_font, bg="#9a7fdd", fg="white", width=30, height=2, relief="flat").pack(pady=10)

        for text, command in simulator_items[4:]:
            tk.Button(right_frame, text=text, command=command, font=button_font, bg="#9a7fdd", fg="white", width=30, height=2, relief="flat").pack(pady=10)

        # --- Botones Adicionales ---
        tk.Button(left_frame, text="Clasificador Diagnóstico", command=self.open_diagnosis, font=button_font, bg="#8b5ffa", fg="white", width=30, height=2, relief="flat").pack(pady=10)
        tk.Button(right_frame, text="Comparador de Gráficas", command=self.open_comparator, font=button_font, bg="#8b5ffa", fg="white", width=30, height=2, relief="flat").pack(pady=10)
        tk.Button(right_frame, text="Verificar Exactitud", command=self.open_accuracy, font=button_font, bg="#8b5ffa", fg="white", width=30, height=2, relief="flat").pack(pady=10)

        tk.Button(content_frame, text="Salir", command=self.root.destroy, font=button_font, bg="#dc3545", fg="white", width=30, height=2, relief="flat").pack(pady=(20, 10))

    def open_simulator(self, dist_type, params):
        self.main_menu.withdraw()
        SimulatorWindow(self.root, self.main_menu, dist_type, params)

    def open_comparator(self):
        self.main_menu.withdraw()
        ComparatorWindow(self.root, self.main_menu)

    def open_diagnosis(self):
        self.main_menu.withdraw()
        DiagnosisWindow(self.root, self.main_menu)
    
    def open_accuracy(self):
        self.main_menu.withdraw()
        AccuracyWindow(self.root, self.main_menu)

# --- CLASE: comparación de Exactitud ---
class AccuracyWindow:
    def __init__(self, root, main_menu):
        self.root = root
        self.main_menu = main_menu
        self.window = tk.Toplevel(root)
        self.window.title("Verificar Exactitud de Diagnóstico")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_to_menu)

        # --- Variables de Estado ---
        self.df_root = None
        self.df_verify = None
        self.root_filename = ""
        self.verify_filename = ""
        self.accuracy_results = {} # Diccionario para guardar resultados actuales

        # --- Constante Directorio Reportes ---
        self.REPORT_DIR = "Informe de exactitud de diagnosticos"
        try:
            os.makedirs(self.REPORT_DIR, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"No se pudo crear carpeta de reportes:\n{e}")

        # --- Layout Principal ---
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=50, pady=30)
        main_container.pack(fill="both", expand=True)

        tk.Label(main_container, text="Verificación de Exactitud de Modelos", font=("Helvetica", 24, "bold"), bg="#1a1a2e", fg="white").pack(pady=(0, 30))

        # --- 1. BOTONES INFERIORES (Empaquetados PRIMERO para asegurar que se vean) ---
        bottom_nav = tk.Frame(main_container, bg="#1a1a2e")
        bottom_nav.pack(side="bottom", fill="x", pady=(20, 0))

        tk.Button(bottom_nav, text="Regresar al Menú", command=self.go_to_menu, bg="#6c757d", fg="white", font=("Helvetica", 12, "bold"), relief="flat", width=25).pack(side="left")
        tk.Button(bottom_nav, text="Comparar Informes", command=self.open_report_comparator, bg="#9a7fdd", fg="white", font=("Helvetica", 12, "bold"), relief="flat", width=20).pack(side="left", padx=5)
        self.btn_download = tk.Button(bottom_nav, text="Descargar Informe", command=self.generate_report, bg="#ff9800", fg="white", font=("Helvetica", 12, "bold"), relief="flat", width=25, state="disabled")
        self.btn_download.pack(side="right")

        # --- Sección Superior: Carga de Archivos ---
        files_frame = tk.Frame(main_container, bg="#1a1a2e")
        files_frame.pack(fill="x", pady=(0, 20))

        # Frame Archivo Raíz
        root_frame = tk.LabelFrame(files_frame, text="1. Archivo de Origen/Raíz (Variable esperada: 'diagnosis')", bg="#2e2e5c", fg="cyan", font=("Helvetica", 12, "bold"), padx=20, pady=20)
        root_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.lbl_root_status = tk.Label(root_frame, text="Ningún archivo cargado", bg="#2e2e5c", fg="#d1c4e9", font=("Helvetica", 10), wraplength=400)
        self.lbl_root_status.pack(pady=(0, 10))
        tk.Button(root_frame, text="Cargar archivo raíz", command=self.load_root_file, bg="#007bff", fg="white", font=("Helvetica", 11, "bold"), relief="flat").pack()

        # Frame Archivo a Verificar
        verify_frame = tk.LabelFrame(files_frame, text="2. Archivo a Verificar (Variable esperada: 'Diagnóstico')", bg="#2e2e5c", fg="magenta", font=("Helvetica", 12, "bold"), padx=20, pady=20)
        verify_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        self.lbl_verify_status = tk.Label(verify_frame, text="Ningún archivo cargado", bg="#2e2e5c", fg="#d1c4e9", font=("Helvetica", 10), wraplength=400)
        self.lbl_verify_status.pack(pady=(0, 10))
        tk.Button(verify_frame, text="Cargar archivo a verificar", command=self.load_verify_file, bg="#9a7fdd", fg="white", font=("Helvetica", 11, "bold"), relief="flat").pack()

        # --- Botón de Acción Principal ---
        self.btn_compare = tk.Button(main_container, text="3. Iniciar comparación", command=self.run_comparison, bg="#28a745", fg="white", font=("Helvetica", 12, "bold"), relief="flat", state="disabled", height=1)
        self.btn_compare.pack(fill="x", pady=10)

        # --- Sección Inferior: Resultados ---
        results_frame = tk.LabelFrame(main_container, text="Resultados de exactitud", bg="#1a1a2e", fg="white", font=("Helvetica", 14, "bold"), padx=20, pady=20)
        results_frame.pack(fill="both", expand=True, pady=(0, 20))

        self.txt_results = scrolledtext.ScrolledText(results_frame, bg="#2e2e5c", fg="white", font=("Consolas", 12), wrap=tk.WORD, relief="flat")
        self.txt_results.pack(fill="both", expand=True)
        self.txt_results.insert(tk.END, "Cargue ambos archivos e inicie la comparación para ver los resultados...")
        self.txt_results.config(state="disabled")

        # --- Botones Inferiores (Navegación y Descarga) ---
        #bottom_nav = tk.Frame(main_container, bg="#1a1a2e")
        #bottom_nav.pack(fill="x")

        #tk.Button(bottom_nav, text="Regresar al Menú", command=self.go_to_menu, bg="#6c757d", fg="white", font=("Helvetica", 12, "bold"), relief="flat", width=25).pack(side="left")

        #self.btn_download = tk.Button(bottom_nav, text="Descargar Informe", command=self.generate_report, bg="#ff9800", fg="white", font=("Helvetica", 12, "bold"), relief="flat", width=25, state="disabled")
        #self.btn_download.pack(side="right")

    def go_to_menu(self):
        self.window.destroy()
        self.main_menu.deiconify()

    def load_file_generic(self, expected_col):
        filepath = filedialog.askopenfilename(filetypes=(("Excel/CSV", "*.xlsx *.csv"), ("All files", "*.*")))
        if not filepath: return None, None, None
        try:
            filename = os.path.basename(filepath)
            if filepath.endswith('.csv'): df = pd.read_csv(filepath)
            else: df = pd.read_excel(filepath)

            # Normalización de nombres de columnas (quitar espacios extra)
            df.columns = df.columns.str.strip()

            if expected_col not in df.columns:
                messagebox.showerror("Error de Columna", f"El archivo no contiene la columna esperada: '{expected_col}'.\nColumnas encontradas: {list(df.columns)}")
                return None, filename, None
            return df, filename, len(df)
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo:\n{e}")
            return None, None, None

    def load_root_file(self):
        df, filename, rows = self.load_file_generic("diagnosis (M=malignant; B=benign)")
        if df is not None:
            self.df_root = df
            self.root_filename = os.path.splitext(filename)[0]
            self.lbl_root_status.config(text=f"Cargado: {filename}\n({rows} registros)", fg="cyan")
            self.check_ready()

    def load_verify_file(self):
        df, filename, rows = self.load_file_generic("Diagnóstico")
        if df is not None:
            self.df_verify = df
            self.verify_filename = os.path.splitext(filename)[0]
            self.lbl_verify_status.config(text=f"Cargado: {filename}\n({rows} registros)", fg="magenta")
            self.check_ready()

    def check_ready(self):
        if self.df_root is not None and self.df_verify is not None:
            self.btn_compare.config(state="normal", bg="#28a745")
        else:
            self.btn_compare.config(state="disabled", bg="#6c757d")

    def run_comparison(self):
        if self.df_root is None or self.df_verify is None: return

        # 1. Validar longitudes
        len_root = len(self.df_root)
        len_verify = len(self.df_verify)
        n_total = min(len_root, len_verify)

        if len_root != len_verify:
             messagebox.showwarning("Advertencia de Tamaño", f"Los archivos tienen diferente número de filas:\nRaíz: {len_root}\nVerificar: {len_verify}\n\nSe compararán solo las primeras {n_total} filas.")

        try:
            # 2. Extraer series y normalizar para comparación (mayúsculas, sin espacios)
            series_root = self.df_root["diagnosis (M=malignant; B=benign)"].iloc[:n_total].astype(str).str.strip().str.upper()
            series_verify = self.df_verify["Diagnóstico"].iloc[:n_total].astype(str).str.strip().str.upper()

            # 3. Comparación vectorizada dato por dato
            matches_mask = (series_root == series_verify)
            n_matches = matches_mask.sum()
            n_mismatches = n_total - n_matches
            accuracy = (n_matches / n_total) * 100 if n_total > 0 else 0

            # 4. Estadísticas adicionales (opcional pero útil)
            # Suponiendo 'M' es positivo/maligno y 'B' es negativo/benigno para una matriz de confusión simple
            true_pos = ((series_root == 'M') & (series_verify == 'M')).sum()
            true_neg = ((series_root == 'B') & (series_verify == 'B')).sum()
            false_pos = ((series_root == 'B') & (series_verify == 'M')).sum() # Era Benigno, predijo Maligno
            false_neg = ((series_root == 'M') & (series_verify == 'B')).sum() # Era Maligno, predijo Benigno
            
            # --- CÁLCULOS DE SENSIBILIDAD Y ESPECIFICIDAD (NUEVO) ---
            total_malignos_reales = true_pos + false_neg
            total_benignos_reales = true_neg + false_pos

            # Evitar división por cero
            sensibilidad = (true_pos / total_malignos_reales * 100) if total_malignos_reales > 0 else 0
            especificidad = (true_neg / total_benignos_reales * 100) if total_benignos_reales > 0 else 0

            # Guardar resultados para el reporte
            self.accuracy_results = {
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n_total": n_total,
                "n_matches": n_matches,
                "n_mismatches": n_mismatches,
                "accuracy": accuracy,
                "sensibilidad": sensibilidad,
                "especificidad": especificidad,
                "tp": true_pos, "tn": true_neg, "fp": false_pos, "fn": false_neg
            }

            # 5. Mostrar Resultados
            report_text = f"--- RESULTADOS DE EXACTITUD ---\n"
            report_text += f"Fecha comparación:  {self.accuracy_results['date']}\n"
            report_text += f"Archivo raíz:       {self.root_filename}\n"
            report_text += f"Archivo Verificado: {self.verify_filename}\n"
            report_text += "-" * 56 + "\n"
            report_text += f"TOTAL REGISTROS COMPARADOS:. {n_total}\n"
            report_text += f"COINCIDENCIAS (Correctos):.. {n_matches}\n"
            report_text += f"DISCREPANCIAS (Errores):.... {n_mismatches}\n\n"
            report_text += f">>> EXACTITUD:.............. {accuracy:.7f} %\n"
            report_text += "-" * 56 + "\n"

            report_text += "--- PORCENTAJES POR CLASE ---\n"
            report_text += f"1. Sensibilidad (Malignos detectados correctos):\n"
            report_text += f"   >>> {sensibilidad:.4f}%  ({true_pos} de {total_malignos_reales} casos reales)\n\n"
            report_text += f"2. Especificidad (Benignos descartados correctos):\n"
            report_text += f"   >>> {especificidad:.4f}%  ({true_neg} de {total_benignos_reales} casos reales)\n"
            report_text += "-" * 56 + "\n"

            report_text += "Detalle (Asumiendo M=Positivo, B=Negativo):\n"
            report_text += f"Verdadero positivo (M correctos): {true_pos}\n"
            report_text += f"Verdadero negatiovo (B correctos): {true_neg}\n"
            report_text += f"Falso positivo (B predicho como M): {false_pos}\n"
            report_text += f"Falso negativo (M predicho como B): {false_neg}\n"

            self.txt_results.config(state="normal")
            self.txt_results.delete("1.0", tk.END)
            self.txt_results.insert(tk.END, report_text)
            self.txt_results.config(state="disabled")

            self.btn_download.config(state="normal")
            messagebox.showinfo("Comparación completada", f"Exactitud calculada: {accuracy:.3f}%")

        except Exception as e:
             messagebox.showerror("Error en comparación", f"Ocurrió un error al comparar los datos:\n{e}")

    def generate_report(self):
        if not self.accuracy_results: return

        try:
            # 1. Determinar nombre del reporte con secuencia
            # Buscar archivos que empiecen con "reporte_No" en la carpeta
            existing_reports = [f for f in os.listdir(self.REPORT_DIR) if f.startswith("reporte_No") and f.endswith(".txt")]
            next_num = 1
            if existing_reports:
                # Extraer números de los reportes existentes. Formato esperado: reporte_NoX_...
                nums = []
                for f in existing_reports:
                    try:
                        # Extraer la parte entre "No" y el siguiente guion bajo "_"
                        part = f.split("No")[1].split("_")[0]
                        nums.append(int(part))
                    except (IndexError, ValueError):
                        pass # Ignorar archivos que no cumplan el formato exacto
                if nums:
                    next_num = max(nums) + 1

            def clean_name(s): return "".join(c for c in s.replace(" ", "_") if c.isalnum() or c in ('_', '-'))
            safe_root = clean_name(self.root_filename)
            safe_verify = clean_name(self.verify_filename)

            report_filename = f"reporte_No{next_num}_({safe_root})_({safe_verify}).txt"
            report_path = os.path.join(self.REPORT_DIR, report_filename)

            # 2. Obtener el texto actual del widget de resultados
            report_content = self.txt_results.get("1.0", tk.END)

            # 3. Guardar archivo
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            messagebox.showinfo("Informe Guardado", f"Informe guardado exitosamente en:\n{self.REPORT_DIR}\n\nArchivo:\n{report_filename}")

        except Exception as e:
            messagebox.showerror("Error al Guardar Informe", f"No se pudo guardar el archivo de texto:\n{e}")

    def open_report_comparator(self):
        # Ocultamos la ventana actual pero no la destruimos
        self.window.withdraw()
        # Abrimos la nueva ventana pasando 'self' como padre para poder regresar
        ReportComparatorWindow(self.root, self.window)

# --- Clase: Ventana de Diagnóstico ---
class DiagnosisWindow:
    def __init__(self, root, main_menu):
        self.root = root
        self.main_menu = main_menu
        self.window = tk.Toplevel(root)
        self.window.title("Clasificador de diagnóstico (GMM)")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_to_menu)

        self.df = None
        self.data = None
        self.loaded_base_filename = "" # Para guardar el nombre base del archivo cargado

        # --- Layout Principal ---
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        tk.Label(main_container, text="Clasificador de diagnóstico", font=("Helvetica", 20, "bold"), bg="#1a1a2e", fg="white").pack(pady=(0, 20))

        # --- Panel de Control Superior ---
        controls_frame = tk.Frame(main_container, bg="#2e2e5c", padx=15, pady=15)
        controls_frame.pack(fill="x", pady=(0, 20))

        params_frame = tk.Frame(controls_frame, bg="#2e2e5c")
        params_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

        tk.Label(params_frame, text="Parámetros del modelo (obtenidos de EM)", font=("Helvetica", 14), bg="#2e2e5c", fg="white").pack(anchor="w", pady=(0, 10))

        benign_frame = tk.LabelFrame(params_frame, text="Normal benigna (Cian)", bg="#2e2e5c", fg="cyan", font=("Helvetica", 12, "bold"))
        benign_frame.pack(fill="x", pady=5)
        self.benign_entries = self.create_param_entries(benign_frame, ["Media (μ)", "Desv. Std (σ)", "Peso (π)"])

        malignant_frame = tk.LabelFrame(params_frame, text="Normal Maligna (Lila)", bg="#2e2e5c", fg="magenta", font=("Helvetica", 12, "bold"))
        malignant_frame.pack(fill="x", pady=5)
        self.malignant_entries = self.create_param_entries(malignant_frame, ["Media (μ)", "Desv. Std (σ)", "Peso (π)"])

        actions_frame = tk.Frame(controls_frame, bg="#2e2e5c")
        actions_frame.pack(side="right", fill="y")

        btn_font = ("Helvetica", 12, "bold")
        tk.Button(actions_frame, text="1. Cargar archivo (Componentes)", command=self.load_data, bg="#007bff", fg="white", font=btn_font, relief="flat", width=30).pack(pady=5)

        col_frame = tk.Frame(actions_frame, bg="#2e2e5c")
        col_frame.pack(pady=5, fill="x")
        tk.Label(col_frame, text="Var:", bg="#2e2e5c", fg="white").pack(side="left")
        self.column_var = tk.StringVar(value="Cargue archivo")
        self.column_menu = OptionMenu(col_frame, self.column_var, "")
        self.column_menu.config(bg="#1a1a2e", fg="white", highlightthickness=0, width=15)
        self.column_menu.pack(side="right", expand=True, fill="x", padx=5)

        # --- NUEVA FILA DE BOTONES HORIZONTALES ---
        btns_row = tk.Frame(actions_frame, bg="#2e2e5c")
        btns_row.pack(pady=10, fill="x")

        # Botón Localizar (Izquierda)
        self.run_btn = tk.Button(btns_row, text="2. Localizar", command=self.run_diagnosis, bg="#28a745", fg="white", font=btn_font, relief="flat", state="disabled")
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Botón Descargar (Derecha) - NUEVO
        self.download_btn = tk.Button(btns_row, text="Descargar excel", command=self.download_excel, bg="#ff9800", fg="white", font=btn_font, relief="flat", state="disabled")
        self.download_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))
        # -----------------------------------------

        tk.Button(actions_frame, text="Ir a Clasificador Bivariado", command=self.open_bivariate_diagnosis, bg="#9C27B0", fg="white", font=btn_font, relief="flat", width=30).pack(pady=(20, 5))
        tk.Button(actions_frame, text="Regresar al menú", command=self.go_to_menu, bg="#6c757d", fg="white", font=btn_font, relief="flat", width=30).pack(pady=(20, 0))

        # --- Panel de Resultados ---
        results_container = tk.Frame(main_container, bg="#1a1a2e")
        results_container.pack(fill="both", expand=True)

        left_result = tk.LabelFrame(results_container, text="Ubicación cartesiana (Valor -> Punto en curva)", bg="#1a1a2e", fg="white", font=("Helvetica", 12))
        left_result.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cols_left = ("Indice", "Valor PC", "Coord X", "Coord Y (Densidad)")
        self.tree_left = ttk.Treeview(left_result, columns=cols_left, show='headings')
        for col in cols_left:
            self.tree_left.heading(col, text=col)
            self.tree_left.column(col, width=80, anchor="center")
        
        vsb_left = ttk.Scrollbar(left_result, orient="vertical", command=self.tree_left.yview)
        self.tree_left.configure(yscrollcommand=vsb_left.set)
        self.tree_left.pack(side="left", fill="both", expand=True)
        vsb_left.pack(side="right", fill="y")

        right_result = tk.LabelFrame(results_container, text="Diagnóstico final", bg="#1a1a2e", fg="white", font=("Helvetica", 12))
        right_result.pack(side="right", fill="both", expand=True, padx=(10, 0))

        cols_right = ("Indice", "Valor PC", "Diagnóstico", "Clase")
        self.tree_right = ttk.Treeview(right_result, columns=cols_right, show='headings')
        for col in cols_right: self.tree_right.heading(col, text=col)
        self.tree_right.column("Indice", width=50, anchor="center")
        self.tree_right.column("Valor PC", width=100, anchor="center")
        self.tree_right.column("Diagnóstico", width=80, anchor="center")
        self.tree_right.column("Clase", width=150, anchor="center")

        vsb_right = ttk.Scrollbar(right_result, orient="vertical", command=self.tree_right.yview)
        self.tree_right.configure(yscrollcommand=vsb_right.set)
        self.tree_right.pack(side="left", fill="both", expand=True)
        vsb_right.pack(side="right", fill="y")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2e2e5c", foreground="white", fieldbackground="#2e2e5c", borderwidth=0)
        style.map('Treeview', background=[('selected', '#9a7fdd')])
        style.configure("Treeview.Heading", background="#4b4b8f", foreground="white", relief="flat")

    def create_param_entries(self, parent, labels):
        entries = {}
        frame = tk.Frame(parent, bg="#2e2e5c")
        frame.pack(fill="x", padx=10, pady=5)
        for label in labels:
            lbl = tk.Label(frame, text=label, bg="#2e2e5c", fg="#d1c4e9", width=15, anchor="w")
            lbl.pack(side="left")
            entry = tk.Entry(frame, bg="#1a1a2e", fg="white", insertbackground="white", width=10, relief="flat")
            entry.pack(side="left", padx=(0, 15))
            entries[label] = entry
            if "Peso" in label: entry.insert(0, "0.5")
            elif "Desv" in label: entry.insert(0, "1.0")
            else: entry.insert(0, "0.0")
        return entries

    def go_to_menu(self):
        self.window.destroy()
        self.main_menu.deiconify()
    
    def open_bivariate_diagnosis(self):
        self.window.withdraw()
        BivariateDiagnosisWindow(self.root, self.window)

    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=(("Excel/CSV", "*.xlsx *.csv"), ("All files", "*.*")))
        if not filepath: return
        try:
            # Guardar nombre base para el archivo de salida
            filename_with_ext = os.path.basename(filepath)
            self.loaded_base_filename = os.path.splitext(filename_with_ext)[0]

            if filepath.endswith('.csv'): self.df = pd.read_csv(filepath)
            else: self.df = pd.read_excel(filepath)
            
            menu = self.column_menu["menu"]
            menu.delete(0, "end")
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                menu.add_command(label=col, command=lambda v=col: self.on_column_select(v))
            
            if numeric_cols: self.on_column_select(numeric_cols[0])
            else: messagebox.showerror("Error", "No hay columnas numéricas.")
        except Exception as e:
            messagebox.showerror("Error de Carga", f"{e}")

    def on_column_select(self, col_name):
        self.column_var.set(col_name)
        self.data = pd.to_numeric(self.df[col_name], errors='coerce').dropna().values
        self.run_btn.config(state="normal")
        self.download_btn.config(state="disabled") # Deshabilitar descarga hasta volver a correr

    def run_diagnosis(self):
        if self.data is None: return
        for item in self.tree_left.get_children(): self.tree_left.delete(item)
        for item in self.tree_right.get_children(): self.tree_right.delete(item)

        try:
            mu_b = float(self.benign_entries["Media (μ)"].get())
            sigma_b = float(self.benign_entries["Desv. Std (σ)"].get())
            pi_b = float(self.benign_entries["Peso (π)"].get())
            mu_m = float(self.malignant_entries["Media (μ)"].get())
            sigma_m = float(self.malignant_entries["Desv. Std (σ)"].get())
            pi_m = float(self.malignant_entries["Peso (π)"].get())

            for i, val in enumerate(self.data):
                dens_b = pi_b * stats.norm.pdf(val, mu_b, sigma_b)
                dens_m = pi_m * stats.norm.pdf(val, mu_m, sigma_m)
                max_y = max(dens_b, dens_m)
                
                if dens_m > dens_b:
                    diag_code, diag_class, tag = "M", "Maligno (Lila)", "malignant"
                else:
                    diag_code, diag_class, tag = "B", "Benigno (Cian)", "benign"

                self.tree_left.insert("", "end", values=(i+1, val, val, max_y))
                self.tree_right.insert("", "end", values=(i+1, val, diag_code, diag_class), tags=(tag,))

            self.tree_right.tag_configure("malignant", foreground="magenta")
            self.tree_right.tag_configure("benign", foreground="cyan")
            
            # Habilitar botón de descarga tras ejecución exitosa
            self.download_btn.config(state="normal")
            messagebox.showinfo("Éxito", "Diagnóstico localizado completado.")

        except ValueError:
            messagebox.showerror("Error de Parámetros", "Asegúrese de que todos los parámetros sean números válidos.")

    # --- NUEVO MÉTODO: Descargar Excel ---
    def download_excel(self):
        if not self.tree_left.get_children():
            messagebox.showwarning("Sin Datos", "Ejecute el diagnóstico primero.")
            return

        try:
            # 1. Extraer datos de los Treeviews a listas
            data_left = []
            for item_id in self.tree_left.get_children():
                data_left.append(self.tree_left.item(item_id)['values'])
            
            data_right = []
            for item_id in self.tree_right.get_children():
                data_right.append(self.tree_right.item(item_id)['values'])

            # 2. Crear DataFrames
            df_diag = pd.DataFrame(data_right, columns=["Indice", "Valor PC", "Diagnóstico", "Clase"])
            df_cartesian = pd.DataFrame(data_left, columns=["Indice", "Valor PC", "Coord X", "Coord Y (Densidad)"])

            # 3. Generar nombre de archivo con versionado
            def clean_name(s): return "".join(c for c in s.replace(" ", "_") if c.isalnum() or c in ('_', '-'))
            
            base_name = f"{clean_name(self.loaded_base_filename)}_{clean_name(self.column_var.get())}_Diagnostico"
            filename = f"{base_name}.xlsx"

            #CARPETA DE GUARDADO
            save_path = os.path.join(CARPETA_DIAGNOSTICO, filename)

            version = 1
            while os.path.exists(save_path):
                filename = f"{base_name}_V{version}.xlsx"
                save_path = os.path.join(CARPETA_DIAGNOSTICO, filename)
                version += 1

            # 4. Guardar en Excel con múltiples hojas
            # Usamos engine='openpyxl' si está disponible, si no pandas intentará el default.
            with pd.ExcelWriter(save_path) as writer:
                df_diag.to_excel(writer, sheet_name="Diagnostico Final", index=False)
                df_cartesian.to_excel(writer, sheet_name="Ubicación Cartesiana", index=False)

            messagebox.showinfo("Descarga Exitosa", f"Archivo guardado en '{CARPETA_DIAGNOSTICO}':\n{filename}")

        except Exception as e:
            messagebox.showerror("Error al Descargar", f"No se pudo guardar el archivo Excel.\nError: {e}")

# --- Clase: Ventana de comparador de reportes ---
class ReportComparatorWindow:
    def __init__(self, root, parent_window):
        self.root = root
        self.parent_window = parent_window
        self.window = tk.Toplevel(root)
        self.window.title("Comparador de Informes Técnicos")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_back)

        self.REPORT_DIR = "Informe de exactitud de diagnosticos"
        self.selected_files = [] # Lista para mantener los nombres de archivos seleccionados
        self.file_vars = {} # Diccionario para guardar las variables Booleanas

        # --- Layout Principal ---
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        # --- Header ---
        header_frame = tk.Frame(main_container, bg="#1a1a2e")
        header_frame.pack(fill="x", pady=(0, 20))
        
        # CAMBIO 1: Botón Gris (#6c757d) en lugar de rojo
        tk.Button(header_frame, text="Regresar a Verificación", command=self.go_back, 
                  bg="#6c757d", fg="white", font=("Helvetica", 11, "bold"), relief="flat").pack(side="left")
        
        # CAMBIO 2: Título actualizado a Máx. 2
        tk.Label(header_frame, text="Comparador de Informes (Máx. 2)", font=("Helvetica", 20, "bold"), 
                 bg="#1a1a2e", fg="white").pack(side="left", padx=20)

        # --- Cuerpo: Dividido en Izquierda (Visualización) y Derecha (Lista) ---
        body_frame = tk.Frame(main_container, bg="#1a1a2e")
        body_frame.pack(fill="both", expand=True)

        # 1. Panel Derecho: CHECKLIST (Lo ponemos primero para asegurar su espacio)
        right_panel = tk.LabelFrame(body_frame, text="Archivos Disponibles", bg="#2e2e5c", fg="cyan", font=("Helvetica", 12, "bold"), width=300)
        right_panel.pack(side="right", fill="y", padx=(20, 0)) # side="right"
        right_panel.pack_propagate(False) 

        # Canvas y Scrollbar para la lista
        canvas = tk.Canvas(right_panel, bg="#2e2e5c", highlightthickness=0)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="#2e2e5c")

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        # 2. Panel Izquierdo: GRID DE COMPARACIÓN (Ocupa el resto del espacio)
        self.display_frame = tk.Frame(body_frame, bg="#1a1a2e")
        self.display_frame.pack(side="left", fill="both", expand=True) # side="left"
        
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(0, weight=1)

        # Cargar archivos al final
        self.load_file_list()

    def go_back(self):
        self.window.destroy()
        self.parent_window.deiconify()

    def load_file_list(self):
        try:
            if not os.path.exists(self.REPORT_DIR):
                tk.Label(self.scrollable_frame, text="No existe carpeta de reportes.", bg="#2e2e5c", fg="white").pack()
                return

            files = sorted([f for f in os.listdir(self.REPORT_DIR) if f.endswith(".txt")], reverse=True)
            
            if not files:
                tk.Label(self.scrollable_frame, text="No hay reportes guardados.", bg="#2e2e5c", fg="white").pack()
                return

            for f in files:
                # --- LÓGICA DE NOMBRE CORTO ---
                display_name = f # Por defecto el nombre original
                
                # 1. Buscamos el número de reporte (ej: No8)
                match_num = re.search(r"reporte_(No\d+)", f)
                report_num = match_num.group(1) if match_num else ""

                # 2. Buscamos el patrón de Variables (ej: 6Variables)
                # Busca dígitos seguidos de la palabra Variables
                match_vars = re.search(r"(\d+Variables)", f, re.IGNORECASE)
                vars_text = match_vars.group(1) if match_vars else ""

                # Si encontramos ambos datos, creamos el nombre bonito
                if report_num and vars_text:
                    display_name = f"{report_num} - {vars_text}"
                elif len(f) > 30:
                    # Si no encuentra patrón pero es muy largo, lo cortamos
                    display_name = f[:25] + "..."

                # ------------------------------

                var = tk.BooleanVar()
                self.file_vars[f] = var # IMPORTANTE: Usamos 'f' (nombre real) como llave
                
                # Checkbutton: muestra 'display_name' pero la lógica usa 'f'
                cb = tk.Checkbutton(self.scrollable_frame, text=display_name, variable=var, 
                                    command=lambda fname=f: self.on_check(fname),
                                    bg="#2e2e5c", fg="white", selectcolor="#1a1a2e", 
                                    activebackground="#2e2e5c", activeforeground="cyan", 
                                    anchor="w", font=("Consolas", 10))
                cb.pack(fill="x", pady=2)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error leyendo carpeta: {e}")

    def on_check(self, filename):
        var = self.file_vars[filename]
        
        if var.get(): # Se acaba de marcar
            # CAMBIO 3: Limite reducido a 2
            if len(self.selected_files) >= 2:
                var.set(False) # Desmarcar inmediatamente
                messagebox.showwarning("Límite alcanzado", "Solo puedes comparar hasta 2 reportes a la vez.")
                return
            self.selected_files.append(filename)
        else: # Se acaba de desmarcar
            if filename in self.selected_files:
                self.selected_files.remove(filename)
        
        self.update_grid_display()

    def parse_report(self, filepath):
        data = {"Exactitud": "N/A", "Sensibilidad": "N/A", "Especificidad": "N/A", "Total": "N/A"}
        full_content = ""
        
        # Banderas de estado para saber qué estamos buscando
        buscando_sensibilidad = False
        buscando_especificidad = False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                full_content = "".join(lines)
                
                for line in lines:
                    line = line.strip() # Quitar espacios al inicio/final

                    # 1. Buscar Exactitud (suele estar en la misma línea)
                    if "EXACTITUD:" in line:
                        parts = line.split(":")
                        if len(parts) > 1: data["Exactitud"] = parts[-1].strip(" .")

                    # 2. Buscar Total Registros
                    if "TOTAL REGISTROS" in line:
                        parts = line.split(":")
                        if len(parts) > 1: data["Total"] = parts[-1].strip(" .")

                    # 3. Detectar títulos para activar búsqueda en siguiente línea
                    if "Sensibilidad" in line and "correctos" in line:
                        buscando_sensibilidad = True
                        continue # Pasamos a la siguiente línea
                    
                    if "Especificidad" in line and "correctos" in line:
                        buscando_especificidad = True
                        continue

                    # 4. Extraer valores si las banderas están activas y encontramos ">>>"
                    if buscando_sensibilidad and line.startswith(">>>"):
                        # Ejemplo: >>> 95.00% (195 de...)
                        parts = line.replace(">>>", "").strip().split("%")
                        if len(parts) > 0: data["Sensibilidad"] = parts[0].strip() + "%"
                        buscando_sensibilidad = False # Ya lo encontramos, apagar bandera

                    if buscando_especificidad and line.startswith(">>>"):
                        parts = line.replace(">>>", "").strip().split("%")
                        if len(parts) > 0: data["Especificidad"] = parts[0].strip() + "%"
                        buscando_especificidad = False

        except Exception:
            pass
        return data, full_content

    def update_grid_display(self):
        # Limpiar grid
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        # CAMBIO 5: Posiciones solo para 2 columnas (lado a lado)
        positions = [(0, 0), (0, 1)]

        for i, filename in enumerate(self.selected_files):
            if i >= len(positions): break # Seguridad
            r, c = positions[i]
            
            # Contenedor estilo "Card"
            card = tk.Frame(self.display_frame, bg="#2e2e5c", bd=2, relief="groove")
            card.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")
            
            # Parsear info
            filepath = os.path.join(self.REPORT_DIR, filename)
            info, raw_text = self.parse_report(filepath)

            # Título del reporte
            tk.Label(card, text=filename, font=("Helvetica", 10, "bold"), bg="#4b4b8f", fg="white", wraplength=400).pack(fill="x", pady=(0, 10))

            # Grid interno para métricas
            metrics_frame = tk.Frame(card, bg="#2e2e5c")
            metrics_frame.pack(fill="x", padx=10)

            # Función helper para labels
            def add_metric(label, value, color):
                row = tk.Frame(metrics_frame, bg="#2e2e5c")
                row.pack(fill="x", pady=4) # Un poco más de padding vertical ya que hay espacio
                tk.Label(row, text=label, font=("Helvetica", 12), bg="#2e2e5c", fg="#d1c4e9").pack(side="left")
                tk.Label(row, text=value, font=("Helvetica", 14, "bold"), bg="#2e2e5c", fg=color).pack(side="right")

            add_metric("Registros:", info["Total"], "white")
            add_metric("Exactitud:", info["Exactitud"], "#00ff00") # Verde brillante
            tk.Frame(metrics_frame, height=2, bg="grey").pack(fill="x", pady=10) # Separador más visible
            add_metric("Sensibilidad:", info["Sensibilidad"], "cyan")
            add_metric("Especificidad:", info["Especificidad"], "magenta")

            # Área de texto scroll
            tk.Label(card, text="Detalle completo:", font=("Helvetica", 10), bg="#2e2e5c", fg="grey").pack(anchor="w", padx=10, pady=(20, 0))
            txt_detail = scrolledtext.ScrolledText(card, bg="#1a1a2e", fg="#d1c4e9", font=("Consolas", 9), relief="flat")
            txt_detail.pack(fill="both", expand=True, padx=10, pady=10)
            txt_detail.insert("1.0", raw_text)
            txt_detail.config(state="disabled")

# --- Clase: Ventana de comparador de reportes ---
class ComparatorWindow:
    def __init__(self, root, main_menu):
        self.root = root
        self.main_menu = main_menu
        self.window = tk.Toplevel(root)
        self.window.title("Comparador de Gráficas EM")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_to_menu)

        self.image_files = []
        self.img_ref_left = None  # Referencia para evitar garbage collector
        self.img_ref_right = None # Referencia para evitar garbage collector

        # Fonts
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.list_font = font.Font(family="Consolas", size=10)

        # --- Layout Principal ---
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=10, pady=10)
        main_container.pack(fill="both", expand=True)

        # Frame superior para el título y el botón de regreso
        top_frame = tk.Frame(main_container, bg="#1a1a2e")
        top_frame.pack(side="top", fill="x", pady=10)

        tk.Label(top_frame, text="Comparador de Gráficas Guardadas", font=self.title_font, bg="#1a1a2e", fg="white").pack(side="left", padx=20)
        tk.Button(top_frame, text="Regresar al Menú", command=self.go_to_menu, bg="#6c757d", fg="white", relief="flat", font=("Helvetica", 10, "bold"), padx=10, pady=5).pack(side="right", padx=20)
        tk.Button(top_frame, text="Actualizar Lista", command=self.populate_lists, bg="#007bff", fg="white", relief="flat", font=("Helvetica", 10, "bold"), padx=10, pady=5).pack(side="right")


        # Frame de contenido para los dos paneles
        content_frame = tk.Frame(main_container, bg="#1a1a2e")
        content_frame.pack(fill="both", expand=True, pady=10)

        # --- Panel Izquierdo ---
        left_panel = tk.Frame(content_frame, bg="#2e2e5c", padx=10, pady=10)
        left_panel.pack(side="left", fill="both", expand=True, padx=10)

        tk.Label(left_panel, text="Seleccionar Gráfica Izquierda:", font=self.title_font, bg="#2e2e5c", fg="white").pack(pady=5)

        list_frame_left = tk.Frame(left_panel, bg="#2e2e5c")
        list_frame_left.pack(fill="x", pady=5)

        scrollbar_left = tk.Scrollbar(list_frame_left, orient="vertical")
        self.listbox_left = tk.Listbox(list_frame_left, yscrollcommand=scrollbar_left.set, bg="#1a1a2e", fg="white", height=5, font=self.list_font, relief="flat", exportselection=False)
        scrollbar_left.config(command=self.listbox_left.yview)
        scrollbar_left.pack(side="right", fill="y")
        self.listbox_left.pack(side="left", fill="x", expand=True)
        self.listbox_left.bind("<<ListboxSelect>>", self.on_select_left)

        self.label_left = tk.Label(left_panel, text="Imagen Izquierda", bg="#1a1a2e", fg="grey", font=self.title_font, height=20) # Height provisional
        self.label_left.pack(fill="both", expand=True, pady=10)

        # --- Panel Derecho ---
        right_panel = tk.Frame(content_frame, bg="#2e2e5c", padx=10, pady=10)
        right_panel.pack(side="right", fill="both", expand=True, padx=10)

        tk.Label(right_panel, text="Seleccionar Gráfica Derecha:", font=self.title_font, bg="#2e2e5c", fg="white").pack(pady=5)

        list_frame_right = tk.Frame(right_panel, bg="#2e2e5c")
        list_frame_right.pack(fill="x", pady=5)

        scrollbar_right = tk.Scrollbar(list_frame_right, orient="vertical")
        self.listbox_right = tk.Listbox(list_frame_right, yscrollcommand=scrollbar_right.set, bg="#1a1a2e", fg="white", height=5, font=self.list_font, relief="flat", exportselection=False)
        scrollbar_right.config(command=self.listbox_right.yview)
        scrollbar_right.pack(side="right", fill="y")
        self.listbox_right.pack(side="left", fill="x", expand=True)
        self.listbox_right.bind("<<ListboxSelect>>", self.on_select_right)

        self.label_right = tk.Label(right_panel, text="Imagen Derecha", bg="#1a1a2e", fg="grey", font=self.title_font, height=20)
        self.label_right.pack(fill="both", expand=True, pady=10)

        self.populate_lists()
        # Bind para reescalar imágenes si la ventana cambia de tamaño (aunque es fullscreen)
        self.label_left.bind("<Configure>", lambda e: self.on_select_left(None))
        self.label_right.bind("<Configure>", lambda e: self.on_select_right(None))


    def go_to_menu(self):
        self.window.destroy()
        self.main_menu.deiconify()

    def populate_lists(self):
        """Carga la lista de imágenes .png desde la carpeta de la galería."""
        try:
            self.image_files = sorted([f for f in os.listdir(GALLERY_DIR) if f.endswith(".png")])
        except FileNotFoundError:
            messagebox.showerror("Error", f"La carpeta '{GALLERY_DIR}' no fue encontrada.")
            self.image_files = []
        except Exception as e:
            messagebox.showerror("Error al leer Galería", f"Error: {e}")
            self.image_files = []

        # Limpiar listas
        self.listbox_left.delete(0, tk.END)
        self.listbox_right.delete(0, tk.END)

        if not self.image_files:
            msg = "No hay gráficas guardadas"
            self.listbox_left.insert(tk.END, msg)
            self.listbox_right.insert(tk.END, msg)
        else:
            for f in self.image_files:
                self.listbox_left.insert(tk.END, f)
                self.listbox_right.insert(tk.END, f)

    def on_select_left(self, event):
        """Maneja la selección de la lista izquierda."""
        if not self.listbox_left.curselection():
            # Esto puede pasar si llamamos desde <Configure> sin selección
            if self.img_ref_left: # Si ya hay una imagen, recargarla
                self.load_image_to_label(self.img_ref_left.filename, self.label_left, "left")
            return

        selected_index = self.listbox_left.curselection()[0]
        filename = self.listbox_left.get(selected_index)
        self.load_image_to_label(filename, self.label_left, "left")

    def on_select_right(self, event):
        """Maneja la selección de la lista derecha."""
        if not self.listbox_right.curselection():
            if self.img_ref_right:
                self.load_image_to_label(self.img_ref_right.filename, self.label_right, "right")
            return

        selected_index = self.listbox_right.curselection()[0]
        filename = self.listbox_right.get(selected_index)
        self.load_image_to_label(filename, self.label_right, "right")

    def load_image_to_label(self, filename, label, side):
        """Carga una imagen por su nombre y la muestra en un Label."""
        if not filename or filename == "No hay gráficas guardadas" or not os.path.exists(os.path.join(GALLERY_DIR, filename)):
            label.config(text="Seleccione una imagen", image='')
            if side == "left": self.img_ref_left = None
            else: self.img_ref_right = None
            return

        try:
            img_path = os.path.join(GALLERY_DIR, filename)

            # Obtener el tamaño del contenedor (Label)
            label.update_idletasks() # Asegurarse de que el tamaño esté actualizado
            max_w = label.winfo_width()
            max_h = label.winfo_height()

            # Evitar tamaños de 1x1 al inicio
            if max_w < 50: max_w = 600
            if max_h < 50: max_h = 600

            img = Image.open(img_path)
            # Redimensionar la imagen para que quepa en el label (thumbnail)
            img.thumbnail((max_w - 10, max_h - 10), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img)

            # Actualizar label
            label.config(image=img_tk, text="")

            # Guardar referencia para evitar que el garbage collector la borre
            if side == "left":
                self.img_ref_left = img_tk
                self.img_ref_left.filename = filename # Guardar el nombre
            else:
                self.img_ref_right = img_tk
                self.img_ref_right.filename = filename # Guardar el nombre

        except Exception as e:
            label.config(text=f"Error al cargar:\n{filename}\n{e}", image='')
            if side == "left": self.img_ref_left = None
            else: self.img_ref_right = None

# --- CLASES simulatorWindow ---
class SimulatorWindow:
    def __init__(self, root, main_menu, dist_type, params):
        self.root = root
        self.main_menu = main_menu
        self.dist_type = dist_type
        self.params = params
        self.df = None
        self.data = None
        self.is_function_shown = False
        self.show_final_mixture = False
        self.em_results = None
        self.em_history = []
        self.em_iteration_vars = []
        self.loaded_filename = "" 

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

        button_frame_bottom = tk.Frame(controls_frame, bg="#2e2e5c")
        button_frame_bottom.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Button(button_frame_bottom, text="Limpiar", command=self.clear_fields, bg="#f44336", fg="white", relief="flat").pack(fill="x", pady=2)
        tk.Button(button_frame_bottom, text="Regresar al Menú", command=self.go_to_menu, bg="#6c757d", fg="white", relief="flat").pack(fill="x", pady=2)

        top_controls_frame = tk.Frame(controls_frame, bg="#2e2e5c")
        top_controls_frame.pack(side="top", fill="both", expand=True)

        tk.Label(top_controls_frame, text=f"Simulador {dist_type}", font=("Helvetica", 18, "bold"), bg="#2e2e5c", fg="white").pack(pady=10)

        desc_label = tk.Label(top_controls_frame, text=DIST_DESCRIPTIONS[dist_type], justify="left", wraplength=300, bg="#2e2e5c", fg="#d1c4e9")
        desc_label.pack(pady=15, padx=5)

        self.entries = {}
        if self.dist_type != "Función Particular":
            for param in self.params:
                row = tk.Frame(top_controls_frame, bg="#2e2e5c")
                tk.Label(row, text=f"{param}:", width=15, anchor='w', bg="#2e2e5c", fg="white").pack(side="left")
                entry = tk.Entry(row, width=10, bg="#1a1a2e", fg="white", insertbackground="white", relief="flat")
                entry.pack(side="right", padx=5)
                row.pack(pady=5, fill="x")
                self.entries[param] = entry
                if param == "Número de Iteraciones":
                    entry.insert(0, "2")

        if self.dist_type != "Algoritmo EM (Cáncer)":
            row = tk.Frame(top_controls_frame, bg="#2e2e5c")
            tk.Label(row, text="Tamaño Muestra:", width=15, anchor='w', bg="#2e2e5c", fg="white").pack(side="left")
            self.size_entry = tk.Entry(row, width=10, bg="#1a1a2e", fg="white", insertbackground="white", relief="flat")
            self.size_entry.insert(0, "1000")
            self.size_entry.pack(side="right", padx=5)
            row.pack(pady=5, fill="x")

        # --- Botones de acción y control ---
        if self.dist_type == "Algoritmo EM (Cáncer)":
            action_em_frame = tk.Frame(top_controls_frame, bg="#2e2e5c")
            action_em_frame.pack(pady=10, fill="x") 
            tk.Button(action_em_frame, text="Bivariado", command=self.open_bivariate_window, bg="#9C27B0", fg="white", font=("Helvetica", 10, "bold"), relief="flat").pack(side="left", expand=True, fill="x", padx=(0, 5))
            tk.Button(action_em_frame, text="1. Cargar Archivo", command=self.load_em_data, bg="#007bff", fg="white", relief="flat").pack(side="right", expand=True, fill="x", padx=(5, 0))
            column_selection_frame = tk.Frame(top_controls_frame, bg="#2e2e5c")
            column_selection_frame.pack(pady=10, fill="x")

            tk.Label(column_selection_frame, text="2. Variable:", anchor='w', bg="#2e2e5c", fg="white").pack(side="left", padx=5) 

            self.em_column_var = tk.StringVar(self.window)
            self.em_column_var.set("Cargue archivo") 
            self.em_column_var.trace("w", self.on_column_change)

            self.column_menu = OptionMenu(column_selection_frame, self.em_column_var, "")
            self.column_menu.config(bg="#1a1a2e", fg="grey", activebackground="#9a7fdd", relief="flat", highlightthickness=0, state="disabled", width=15) 
            self.column_menu["menu"].config(bg="#1a1a2e", fg="white")
            self.column_menu.pack(side="left", expand=True, fill="x", padx=5)

            run_save_frame = tk.Frame(top_controls_frame, bg="#2e2e5c")
            run_save_frame.pack(pady=5, fill="x") 

            self.run_em_button = tk.Button(run_save_frame, text="3. Ejecutar EM", command=self.run_em_algorithm, bg="#4CAF50", fg="white", relief="flat", state="disabled")
            self.run_em_button.pack(side="left", padx=(0,5), fill="x", expand=True) 

            self.save_plot_button = tk.Button(run_save_frame, text="4. Guardar Gráfica", command=self.save_plot, bg="#ff9800", fg="white", relief="flat", state="disabled")
            self.save_plot_button.pack(side="left", padx=(5,0), fill="x", expand=True) 

            history_frame = tk.Frame(top_controls_frame, bg="#2e2e5c")
            history_frame.pack(pady=5, fill="x") 
            self.history_button = tk.Button(history_frame, text="Ver Historial", command=self.show_history_window, bg="#6c757d", fg="white", relief="flat", state="disabled")
            self.history_button.pack(side="left", padx=(0,5), fill="x", expand=True) 
            tk.Button(history_frame, text="Limpiar Historial", command=self.clear_em_history, bg="#f44336", fg="white", relief="flat").pack(side="left", padx=(5,0), fill="x", expand=True) 

            tk.Button(top_controls_frame, text="Superponer Mezcla Final", command=self.toggle_final_mixture, bg="#9a7fdd", fg="white", relief="flat").pack(fill="x", pady=5)

            self.show_final_params = tk.BooleanVar(value=True)
            self.final_params_checkbox = tk.Checkbutton(top_controls_frame, text="Mostrar Parámetros Finales", variable=self.show_final_params, command=self.draw_plot, bg="#2e2e5c", fg="white", selectcolor="#1a1a2e", anchor='w', state="disabled")
            self.final_params_checkbox.pack(fill="x", pady=5, padx=5)

            self.show_all_equations_button = tk.Button(top_controls_frame, text="Ver Ecuaciones", command=self.show_all_equations_window, bg="#ff9800", fg="white", relief="flat", state="disabled")
            self.show_all_equations_button.pack(fill="x", pady=5)

            checklist_container = tk.Frame(top_controls_frame, bg="#2e2e5c")
            checklist_container.pack(fill="both", expand=True, pady=10)
            tk.Label(checklist_container, text="Superponer Iteraciones:", font=("Helvetica", 12), bg="#2e2e5c", fg="white").pack(anchor='w') 

            checklist_canvas = tk.Canvas(checklist_container, bg="#1a1a2e", highlightthickness=0)
            checklist_scrollbar = tk.Scrollbar(checklist_container, orient="vertical", command=checklist_canvas.yview)
            self.checklist_frame = tk.Frame(checklist_canvas, bg="#1a1a2e")

            self.checklist_frame.bind("<Configure>", lambda e: checklist_canvas.configure(scrollregion=checklist_canvas.bbox("all")))
            checklist_canvas.create_window((0, 0), window=self.checklist_frame, anchor="nw")
            checklist_canvas.configure(yscrollcommand=checklist_scrollbar.set)

            checklist_canvas.pack(side="left", fill="both", expand=True)
            checklist_scrollbar.pack(side="right", fill="y")

        else: # Otros simuladores
            action_button_frame = tk.Frame(top_controls_frame, bg="#2e2e5c")
            action_button_frame.pack(pady=10)
            tk.Button(action_button_frame, text="Simular", command=self.run_simulation, bg="#4CAF50", fg="white", relief="flat").pack(side="left", padx=5)

            self.toggle_func_btn = tk.Button(action_button_frame, text="Superponer Función", command=self.toggle_theory_function, bg="#9a7fdd", fg="white", relief="flat")
            self.toggle_func_btn.pack(side="left", padx=5)

            if self.dist_type in ["Normal Bivariada", "Función Particular"]:
                self.btn_3d = tk.Button(action_button_frame, text="Visualizar en 3D", command=self.show_3d_plot, bg="#ff9800", fg="white", relief="flat")
                self.btn_3d.pack(side="left", padx=5)

        if self.dist_type == "Normal Bivariada":
            tk.Label(top_controls_frame, text="Muestra Generada:", font=("Helvetica", 12), bg="#2e2e5c", fg="white").pack(pady=(10, 5), anchor='w')
            self.results_button = tk.Button(top_controls_frame, text="Ver Muestra en Nueva Ventana", command=self.show_results_window, bg="#6c757d", fg="white", relief="flat", state="disabled")
            self.results_button.pack(fill="x", pady=5)
        elif self.dist_type != "Algoritmo EM (Cáncer)":
            tk.Label(top_controls_frame, text="Resultados:", font=("Helvetica", 12), bg="#2e2e5c", fg="white").pack(pady=(10, 5), anchor='w') 
            self.data_text = scrolledtext.ScrolledText(top_controls_frame, height=10, width=35, bg="#1a1a2e", fg="white", relief="flat") 
            self.data_text.pack(fill="both", expand=True, pady=(5, 0))

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
        self.df = None
        self.data = None
        self.em_results = None
        self.em_history = []
        self.show_final_mixture = False
        
        if hasattr(self, 'data_text'):
            self.data_text.delete('1.0', tk.END)
        if hasattr(self, 'results_button'):
            self.results_button.config(state="disabled")
        if self.dist_type == "Algoritmo EM (Cáncer)":
            self.em_column_var.set("Cargue archivo") 
            self.column_menu.config(state="disabled", fg="grey")
            if hasattr(self, 'show_final_params'):
                self.show_final_params.set(True)
            if hasattr(self, 'final_params_checkbox'):
                self.final_params_checkbox.config(state='disabled')
            if hasattr(self, 'show_all_equations_button'):
                self.show_all_equations_button.config(state="disabled")
            if hasattr(self, 'save_plot_button'):
                self.save_plot_button.config(state="disabled")

            self.run_em_button.config(state="disabled")
            self.history_button.config(state="disabled")
            if hasattr(self, 'checklist_frame'):
                for widget in self.checklist_frame.winfo_children():
                    widget.destroy()
            self.em_iteration_vars = []

        self.ax.clear()
        self.ax.set_title("El histograma aparecerá aquí")
        self.canvas.draw()

    def go_to_menu(self):
        self.window.destroy()
        self.main_menu.deiconify()

    # --- MÉTODO NUEVO EN SimulatorWindow ---
    def open_bivariate_window(self):
        # Si no hay datos cargados, avisamos, pero permitimos abrir por si quieren cargar allá
        if self.df is None:
            messagebox.showinfo("Aviso", "Se recomienda cargar un archivo primero, o puede cargarlo en la siguiente ventana.")
            return
        # Ocultamos la ventana actual
        self.window.withdraw()
        # Abrimos la nueva ventana pasando el root, esta ventana (self.window) y el dataframe actual
        BivariateEMWindow(self.root, self.window, self.df)
    
    def load_em_data(self):
        try:
            filepath = filedialog.askopenfilename(
                title="Selecciona el archivo de datos",
                filetypes=(("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if not filepath: return

            filename_with_ext = os.path.basename(filepath)
            new_loaded_filename = os.path.splitext(filename_with_ext)[0]
            
            self.clear_fields() 
            
            self.loaded_filename = new_loaded_filename
            print(f"Archivo cargado: '{self.loaded_filename}'") 

            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath)
            else:
                self.df = pd.read_excel(filepath)

            menu = self.column_menu["menu"]
            menu.delete(0, "end")
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            suggested_columns = [
                'radius (nucA)', 'texture (nucA)', 'perimeter (nucA)', 'area (nucA)',
                'smoothness (nucA)', 'compactness (nucA)', 'concavity (nucA)', 'concave points (nucA)',
                'symmetry (nucA)', 'fractal dimension (nucA)',
                'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'
            ]
            available_columns = [col for col in suggested_columns if col in numeric_cols]
            other_numeric_cols = [col for col in numeric_cols if col not in available_columns]
            available_columns.extend(other_numeric_cols)

            if not available_columns:
                messagebox.showerror("Error de Datos", "No se encontraron columnas numéricas.")
                return

            for col in available_columns:
                menu.add_command(label=col, command=lambda value=col: self.em_column_var.set(value))

            self.em_column_var.set(available_columns[0])
            self.column_menu.config(state="normal", fg="white")

        except Exception as e:
            messagebox.showerror("Error al Cargar Datos", f"Ocurrió un error: {e}")

    def on_column_change(self, *args):
        if self.df is None: return
        self.em_results = None
        self.show_final_mixture = False
        self.final_params_checkbox.config(state='disabled')
        self.show_all_equations_button.config(state="disabled")
        self.save_plot_button.config(state="disabled")

        if hasattr(self, 'checklist_frame'):
            for widget in self.checklist_frame.winfo_children(): widget.destroy()
        self.em_iteration_vars = []

        column = self.em_column_var.get()
        if not column or column == "Cargue archivo": return

        try:
            numeric_data = pd.to_numeric(self.df[column], errors='coerce')
            self.data = numeric_data.dropna().values

            if len(self.data) == 0:
                messagebox.showwarning("Datos Vacíos", f"La columna '{column}' está vacía.")
                self.run_em_button.config(state="disabled")
                self.ax.clear(); self.ax.set_title("Datos no válidos"); self.canvas.draw()
                return

            self.run_em_button.config(state="normal")
            self.draw_plot()

        except KeyError:
             messagebox.showerror("Error de Columna", f"Columna '{column}' no encontrada.")
             self.run_em_button.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"No se pudo procesar '{column}'.\nError: {e}")
            self.run_em_button.config(state="disabled")

    def run_em_algorithm(self):
        try:
            if self.data is None: messagebox.showinfo("Información", "No hay datos cargados."); return

            if hasattr(self, 'checklist_frame'):
                for widget in self.checklist_frame.winfo_children(): widget.destroy()
            self.em_iteration_vars = []

            num_iterations = int(self.entries["Número de Iteraciones"].get())
            if num_iterations <= 0: raise ValueError("Iteraciones debe ser > 0.")

            n_total = len(self.data)
            data_sorted = np.sort(self.data)
            n1 = n_total // 2
            data1, data2 = data_sorted[:n1], data_sorted[n1:]

            mu1_init = np.mean(data1) if len(data1) > 0 else np.mean(self.data) - np.std(self.data)
            mu2_init = np.mean(data2) if len(data2) > 0 else np.mean(self.data) + np.std(self.data)
            sigma1_init = np.std(data1) if len(data1) > 1 else np.std(self.data)
            sigma2_init = np.std(data2) if len(data2) > 1 else np.std(self.data)

            params = {'pi1': 0.5, 'pi2': 0.5, 'mu1': mu1_init, 'mu2': mu2_init,
                      'sigma1': max(sigma1_init, 0.01), 'sigma2': max(sigma2_init, 0.01)}
            iteration_steps = []

            for i in range(num_iterations):
                pdf1 = stats.norm.pdf(self.data, params['mu1'], params['sigma1'])
                pdf2 = stats.norm.pdf(self.data, params['mu2'], params['sigma2'])
                numerator1 = params['pi1'] * pdf1; numerator2 = params['pi2'] * pdf2
                denominator = numerator1 + numerator2
                denominator[denominator == 0] = np.finfo(float).eps
                gamma1 = numerator1 / denominator; gamma2 = 1 - gamma1
                sum_gamma1, sum_gamma2 = np.sum(gamma1), np.sum(gamma2)

                if sum_gamma1 < 1e-6 or sum_gamma2 < 1e-6:
                    print(f"Advertencia: Componente colapsó en iteración {i+1}.")
                    break

                params['pi1'], params['pi2'] = sum_gamma1 / n_total, sum_gamma2 / n_total
                params['mu1'], params['mu2'] = np.sum(gamma1 * self.data) / sum_gamma1, np.sum(gamma2 * self.data) / sum_gamma2
                params['sigma1'] = np.sqrt(np.sum(gamma1 * (self.data - params['mu1']) ** 2) / sum_gamma1)
                params['sigma2'] = np.sqrt(np.sum(gamma2 * (self.data - params['mu2']) ** 2) / sum_gamma2)
                iteration_steps.append(params.copy())

            self.em_results = {"steps": iteration_steps}
            history_entry = {"column": self.em_column_var.get(), "iterations": num_iterations, "steps": iteration_steps}
            self.em_history.append(history_entry)
            self.history_button.config(state="normal")

            for i in range(len(iteration_steps)):
                iter_frame = tk.Frame(self.checklist_frame, bg="#2e2e5c")
                tk.Label(iter_frame, text=f"Iter {i+1}:", fg="white", bg="#2e2e5c").pack(side="left", padx=5) 
                var1 = tk.BooleanVar(); cb1 = tk.Checkbutton(iter_frame, text="N1", variable=var1, command=self.draw_plot, bg="#2e2e5c", fg="cyan", selectcolor="#1a1a2e"); cb1.pack(side="left")
                var2 = tk.BooleanVar(); cb2 = tk.Checkbutton(iter_frame, text="N2", variable=var2, command=self.draw_plot, bg="#2e2e5c", fg="magenta", selectcolor="#1a1a2e"); cb2.pack(side="left")
                var_mezcla = tk.BooleanVar(); cb_mezcla = tk.Checkbutton(iter_frame, text="Mezcla", variable=var_mezcla, command=self.draw_plot, bg="#2e2e5c", fg="yellow", selectcolor="#1a1a2e"); cb_mezcla.pack(side="left")
                var_params = tk.BooleanVar(value=True); cb_params = tk.Checkbutton(iter_frame, text="Val", variable=var_params, command=self.draw_plot, bg="#2e2e5c", fg="#FFD700", selectcolor="#1a1a2e"); cb_params.pack(side="left", padx=5) 
                self.em_iteration_vars.append((var1, var2, var_mezcla, var_params))
                iter_frame.pack(anchor="w", fill="x") 

            self.show_all_equations_button.config(state="normal")
            self.save_plot_button.config(state="normal")
            self.draw_plot()

        except Exception as e: messagebox.showerror("Error en Algoritmo EM", f"Ocurrió un error: {e}")

    def save_plot(self):
        if self.df is None or self.fig is None:
            messagebox.showwarning("Sin Gráfica", "Ejecute el algoritmo primero.")
            return
        try:
            variable = self.em_column_var.get()
            iters = self.entries["Número de Iteraciones"].get()
            current_filename = self.loaded_filename if self.loaded_filename else "datos"
            def clean_filename(s):
                s = s.replace(" ", "_")
                return "".join(c for c in s if c.isalnum() or c in ('_', '-', '(', ')'))

            safe_filename = clean_filename(current_filename)
            safe_variable = clean_filename(variable)
            safe_iters = "".join(c for c in iters if c.isdigit())

            base_name = f"{safe_filename}_{safe_variable}_{safe_iters}iters"
            filename = f"{base_name}.png"
            save_path = os.path.join(GALLERY_DIR, filename)

            version = 1
            while os.path.exists(save_path):
                filename = f"{base_name}_V{version}.png"
                save_path = os.path.join(GALLERY_DIR, filename)
                version += 1

            self.fig.patch.set_facecolor('#1a1a2e')
            self.fig.savefig(save_path, facecolor=self.fig.get_facecolor(), dpi=150, bbox_inches='tight')
            messagebox.showinfo("Éxito", f"Gráfica guardada como:\n{filename}")

        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar: {e}")

    def run_simulation(self):
        try:
            params_values = {p: float(e.get()) for p, e in self.entries.items()}
            size = int(self.size_entry.get())
            if size <= 0: raise ValueError("Tamaño > 0.")

            if self.dist_type == "Bernoulli":
                p = params_values["Probabilidad (p)"]; self.data = np.random.binomial(1, p, size)
            elif self.dist_type == "Binomial":
                n, p = int(params_values["Ensayos (n)"]), params_values["Probabilidad (p)"]; self.data = np.random.binomial(n, p, size)
            elif self.dist_type == "Exponencial":
                lambda_ = params_values["Tasa (λ)"]; self.data = np.random.exponential(scale=1 / lambda_, size=size)
            elif self.dist_type == "Normal":
                mu, sigma = params_values["Media (μ)"], params_values["Desv. Estándar (σ)"]; self.data = np.random.normal(mu, sigma, size)
            elif self.dist_type == "Función Particular":
                samples = []; x, y = 1.0, 1.0; burn_in = 500; total_iter = size + burn_in
                for i in range(total_iter):
                    u1 = np.random.uniform(0, 1); b_x = 3*y + 2; c_x = -u1*(8 + 6*y); x = (-b_x + np.sqrt(b_x**2 - 4*c_x)) / 2
                    u2 = np.random.uniform(0, 1); b_y = 2*x + 2; c_y = -u2*(4*x + 10); y = (-b_y + np.sqrt(b_y**2 - 6*c_y)) / 3
                    if i >= burn_in: samples.append([x, y])
                self.data = np.array(samples)
            elif self.dist_type == "Normal Bivariada":
                mu_x, mu_y, sigma_x, sigma_y, rho = (params_values["μ_x"], params_values["μ_y"], params_values["σ_x"], params_values["σ_y"], params_values["Correlación (ρ)"])
                samples, x, y, burn_in = np.zeros((size, 2)), 0.0, 0.0, 500
                for i in range(size + burn_in):
                    mu_cond_x = mu_x + rho * (sigma_x / sigma_y) * (y - mu_y); sigma_cond_x = np.sqrt(sigma_x**2 * (1 - rho**2)); x = np.random.normal(mu_cond_x, sigma_cond_x)
                    mu_cond_y = mu_y + rho * (sigma_y / sigma_x) * (x - mu_x); sigma_cond_y = np.sqrt(sigma_y**2 * (1 - rho**2)); y = np.random.normal(mu_cond_y, sigma_cond_y)
                    if i >= burn_in: samples[i - burn_in] = [x, y]
                self.data = samples

            self.is_function_shown = False; self.draw_plot()
            if self.dist_type == "Normal Bivariada": self.results_button.config(state="normal")
            elif hasattr(self, 'data_text'):
                self.data_text.delete('1.0', tk.END)
                if self.dist_type == "Bernoulli":
                    successes = np.sum(self.data); total = len(self.data); summary_str = f"Éxitos (1): {successes}\nFracasos (0): {total - successes}"
                    self.data_text.insert(tk.END, summary_str)
                elif self.dist_type == "Binomial":
                    n = int(params_values["Ensayos (n)"]); avg_successes = np.mean(self.data); summary_str = f"Promedio éxitos: {avg_successes:.2f}"
                    self.data_text.insert(tk.END, summary_str)
                else: self.data_text.insert(tk.END, np.array2string(self.data, precision=4, separator=', ', max_line_width=30))
        except (ValueError, KeyError) as e: messagebox.showerror("Error", f"Revise parámetros.\nError: {e}")
        except Exception as e: messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def toggle_theory_function(self):
        if self.data is None: messagebox.showinfo("Información", "Simule primero."); return
        self.is_function_shown = not self.is_function_shown; self.draw_plot()

    def toggle_final_mixture(self):
        if self.em_results is None: messagebox.showinfo("Información", "Ejecute EM primero."); return
        self.show_final_mixture = not self.show_final_mixture
        new_state = "normal" if self.show_final_mixture else "disabled"; self.final_params_checkbox.config(state=new_state)
        self.draw_plot()

    def draw_plot(self):
        if self.data is None: self.ax.clear(); self.ax.set_title("Cargue datos"); self.canvas.draw(); return
        self.ax.clear()
        try: params = {p: float(e.get()) for p, e in self.entries.items()} if self.entries else {}
        except ValueError: params = {}

        if self.dist_type == "Bernoulli":
            p = params.get("Probabilidad (p)", 0.5); bins = np.arange(-0.5, 2.5, 1)
            self.ax.hist(self.data, bins=bins, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown: self.ax.plot([0, 1], [1 - p, p], 'yo', ms=10, ls='--', label="PMF Teórica")
            self.ax.set_xticks([0, 1]); self.ax.set_title(f"Bernoulli (p={p:.2f})", color="white")
        elif self.dist_type == "Binomial":
            n, p = int(params.get("Ensayos (n)", 1)), params.get("Probabilidad (p)", 0.5)
            bins = np.arange(-0.5, n + 1.5, 1)
            self.ax.hist(self.data, bins=bins, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown: x = np.arange(0, n + 1); pmf = stats.binom.pmf(x, n, p); self.ax.plot(x, pmf, 'yo-', label="PMF Teórica")
            self.ax.set_title(f"Binomial (n={n}, p={p:.2f})", color="white")
        elif self.dist_type == "Exponencial":
            lambda_ = params.get("Tasa (λ)", 1)
            self.ax.hist(self.data, bins=30, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown: x = np.linspace(self.data.min(), self.data.max(), 100); pdf = stats.expon.pdf(x, scale=1 / lambda_); self.ax.plot(x, pdf, 'y-', lw=2, label="PDF Teórica")
            self.ax.set_title(f"Exponencial (λ={lambda_:.2f})", color="white")
        elif self.dist_type == "Normal":
            mu, sigma = params.get("Media (μ)", 0), params.get("Desv. Estándar (σ)", 1)
            self.ax.hist(self.data, bins=30, density=True, alpha=0.7, label="Muestra", color="#9a7fdd")
            if self.is_function_shown: x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100); pdf = stats.norm.pdf(x, mu, sigma); self.ax.plot(x, pdf, 'y-', lw=2, label="PDF Teórica")
            self.ax.set_title(f"Normal (μ={mu:.2f}, σ={sigma:.2f})", color="white")
        elif self.dist_type == "Función Particular":
            self.ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.5, s=15, color="#9a7fdd", edgecolor='none')
            self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_xlim(0, 2); self.ax.set_ylim(0, 2)
            if self.is_function_shown: x_grid=np.linspace(0,2,100); y_grid=np.linspace(0,2,100); X,Y=np.meshgrid(x_grid,y_grid); Z=(2*X+3*Y+2)/28; self.ax.contour(X,Y,Z,colors='yellow',alpha=0.7)
            self.ax.set_title("Función Particular", color="white")
        elif self.dist_type == "Normal Bivariada":
            mu_x, mu_y = params.get("μ_x", 0), params.get("μ_y", 0); sigma_x, sigma_y = params.get("σ_x", 1), params.get("σ_y", 1); rho = params.get("Correlación (ρ)", 0)
            self.ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.5, s=15, color="#9a7fdd", edgecolor='none')
            self.ax.set_xlabel("X"); self.ax.set_ylabel("Y")
            if self.is_function_shown: x_grid=np.linspace(mu_x-3*sigma_x,mu_x+3*sigma_x,100); y_grid=np.linspace(mu_y-3*sigma_y,mu_y+3*sigma_y,100); X,Y=np.meshgrid(x_grid,y_grid); pos=np.dstack((X,Y)); cov=[[sigma_x**2,rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,sigma_y**2]]; rv=stats.multivariate_normal([mu_x,mu_y],cov); self.ax.contour(X,Y,rv.pdf(pos),colors='yellow',alpha=0.7)
            self.ax.set_title("Normal Bivariada", color="white")
        elif self.dist_type == "Algoritmo EM (Cáncer)":
            selected_column = self.em_column_var.get(); bins = 50
            self.ax.hist(self.data, bins=bins, density=True, alpha=0.6, label=f"Datos '{selected_column}'", color="#9a7fdd")
            legend_needed = any(v.get() for row in self.em_iteration_vars for v in row[:3]) or self.show_final_mixture
            if self.em_results:
                x_curve = np.linspace(self.data.min(), self.data.max(), 500); num_iterations = len(self.em_iteration_vars)
                colors1 = plt.cm.cool(np.linspace(0.3, 1, num_iterations)); colors2 = plt.cm.autumn(np.linspace(0.3, 1, num_iterations)); colors_mezcla = plt.cm.summer(np.linspace(0.3, 1, num_iterations))
                for i, (var1, var2, var_mezcla, var_params) in enumerate(self.em_iteration_vars):
                    if var1.get() or var2.get() or var_mezcla.get():
                        p = self.em_results['steps'][i]
                        if var1.get():
                            pdf1 = stats.norm.pdf(x_curve, p['mu1'], p['sigma1']) * p['pi1']; self.ax.plot(x_curve, pdf1, color=colors1[i], ls='--', label=f'N1 (Iter {i+1})')
                            if var_params.get(): peak_y1 = np.max(pdf1) if pdf1.size>0 else 0; peak_x1 = x_curve[np.argmax(pdf1)] if pdf1.size>0 else self.data.min(); param_text1 = f"μ₁={p['mu1']:.4f}\nσ₁={p['sigma1']:.4f}\nπ₁={p['pi1']:.4f}"; self.ax.text(peak_x1,peak_y1,param_text1,color=colors1[i],fontsize=8,va='bottom',ha='center',backgroundcolor=(0,0,0,0.5))
                        if var2.get():
                            pdf2 = stats.norm.pdf(x_curve, p['mu2'], p['sigma2']) * p['pi2']; self.ax.plot(x_curve, pdf2, color=colors2[i], ls=':', label=f'N2 (Iter {i+1})')
                            if var_params.get(): peak_y2 = np.max(pdf2) if pdf2.size>0 else 0; peak_x2 = x_curve[np.argmax(pdf2)] if pdf2.size>0 else self.data.max(); param_text2 = f"μ₂={p['mu2']:.4f}\nσ₂={p['sigma2']:.4f}\nπ₂={p['pi2']:.4f}"; self.ax.text(peak_x2,peak_y2,param_text2,color=colors2[i],fontsize=8,va='bottom',ha='center',backgroundcolor=(0,0,0,0.5))
                        if var_mezcla.get():
                            pdf1_m=stats.norm.pdf(x_curve, p['mu1'], p['sigma1'])*p['pi1']; pdf2_m=stats.norm.pdf(x_curve, p['mu2'], p['sigma2'])*p['pi2']; self.ax.plot(x_curve,pdf1_m+pdf2_m,color=colors_mezcla[i],ls='-',label=f'Mezcla (Iter {i+1})')
                if self.show_final_mixture and self.em_results['steps']:
                    p_final = self.em_results['steps'][-1]; pdf1_final = stats.norm.pdf(x_curve, p_final['mu1'], p_final['sigma1']) * p_final['pi1']; pdf2_final = stats.norm.pdf(x_curve, p_final['mu2'], p_final['sigma2']) * p_final['pi2']
                    self.ax.plot(x_curve, pdf1_final + pdf2_final, 'y-', lw=3, label="Mezcla Final"); self.ax.plot(x_curve, pdf1_final, color='cyan', ls='--', lw=1.5, alpha=0.8); self.ax.plot(x_curve, pdf2_final, color='magenta', ls=':', lw=1.5, alpha=0.8)
                    if self.show_final_params.get():
                        peak_y1_final=np.max(pdf1_final) if pdf1_final.size>0 else 0; peak_x1_final=x_curve[np.argmax(pdf1_final)] if pdf1_final.size>0 else self.data.min(); param_text1_final=f"Final\nμ₁={p_final['mu1']:.4f}\nσ₁={p_final['sigma1']:.4f}\nπ₁={p_final['pi1']:.4f}"; self.ax.text(peak_x1_final,peak_y1_final,param_text1_final,color='cyan',fontsize=9,va='bottom',ha='center',backgroundcolor=(0,0,0,0.5))
                        peak_y2_final=np.max(pdf2_final) if pdf2_final.size>0 else 0; peak_x2_final=x_curve[np.argmax(pdf2_final)] if pdf2_final.size>0 else self.data.max(); param_text2_final=f"Final\nμ₂={p_final['mu2']:.4f}\nσ₂={p_final['sigma2']:.4f}\nπ₂={p_final['pi2']:.4f}"; self.ax.text(peak_x2_final,peak_y2_final,param_text2_final,color='magenta',fontsize=9,va='bottom',ha='center',backgroundcolor=(0,0,0,0.5))
            self.ax.set_title("Resultado Algoritmo EM", color="white"); self.ax.set_xlabel(f"Valor ({selected_column})"); self.ax.set_ylabel("Densidad");
            if legend_needed: self.ax.legend(loc='upper right')
            self.fig.tight_layout()

        if self.dist_type not in ["Normal Bivariada", "Función Particular", "Algoritmo EM (Cáncer)"]:
            if self.is_function_shown: self.ax.legend(); self.ax.set_xlabel("Valor"); self.ax.set_ylabel("Densidad / Probabilidad")
        self.canvas.draw()

    def show_results_window(self):
        if self.data is None: messagebox.showinfo("Información", "Simule primero."); return
        results_win = tk.Toplevel(self.window); results_win.title(f"Muestra - {self.dist_type}"); results_win.geometry("400x500"); results_win.configure(bg="#1a1a2e")
        tk.Label(results_win, text="Muestra Simulada", font=("Helvetica", 14, "bold"), bg="#1a1a2e", fg="white").pack(pady=10)
        text_area = scrolledtext.ScrolledText(results_win, bg="#2e2e5c", fg="white", relief="flat"); text_area.pack(fill="both", expand=True, padx=10, pady=10)
        data_str = np.array2string(self.data, precision=4, separator=', ', max_line_width=30); text_area.insert(tk.END, data_str); text_area.config(state="disabled")

    def clear_em_history(self): self.em_history = []; self.history_button.config(state="disabled"); messagebox.showinfo("Historial", "Historial limpiado.")

    def show_history_window(self):
        if not self.em_history: messagebox.showinfo("Historial Vacío", "No hay ejecuciones."); return
        history_win = tk.Toplevel(self.window); history_win.title("Historial Ejecuciones EM"); history_win.geometry("1200x700"); history_win.configure(bg="#1a1a2e")
        main_frame = tk.Frame(history_win, bg="#1a1a2e"); main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        bottom_frame = tk.Frame(main_frame, bg="#1a1a2e"); bottom_frame.pack(side="bottom", fill="x", pady=(10, 0))
        def clear_and_close(): self.clear_em_history(); history_win.destroy()
        tk.Button(bottom_frame, text="Limpiar Historial", command=clear_and_close, bg="#f44336", fg="white", relief="flat").pack()
        top_frame = tk.Frame(main_frame, bg="#1a1a2e"); top_frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(top_frame, bg="#1a1a2e", highlightthickness=0)
        v_scrollbar = tk.Scrollbar(top_frame, orient="vertical", command=canvas.yview); v_scrollbar.pack(side="right", fill="y")
        x_scrollbar = tk.Scrollbar(top_frame, orient="horizontal", command=canvas.xview); x_scrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True); canvas.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=v_scrollbar.set)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e"); canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        for i, entry in enumerate(self.em_history):
            exec_num = i + 1; iters = entry['iterations']; col_name = entry.get('column', 'N/A')
            col_frame = tk.Frame(scrollable_frame, bg="#2e2e5c", padx=10, pady=10, bd=1, relief="solid"); col_frame.grid(row=0, column=i, sticky="ns", padx=10, pady=10)
            header = f"Ejec. {exec_num} ({iters} iters)\nVar: {col_name}"; tk.Label(col_frame, text=header, font=("Courier", 12, "bold"), bg="#2e2e5c", fg="white", justify="left").pack(anchor='w', pady=(0, 10)) 
            for j, step_params in enumerate(entry['steps']):
                iter_header = f"-- Iter {j+1} --"; tk.Label(col_frame, text=iter_header, font=("Courier", 10, "underline"), bg="#2e2e5c", fg="white").pack(anchor='w', pady=(5,0)) 
                p = step_params
                tk.Label(col_frame, text=" G1:", font=("Courier", 10, "bold"), bg="#2e2e5c", fg="cyan").pack(anchor='w', pady=(5,0)); tk.Label(col_frame, text=f"  μ:{p['mu1']:.4f}", font=("Courier",10), bg="#2e2e5c", fg="white").pack(anchor='w'); tk.Label(col_frame, text=f"  σ:{p['sigma1']:.4f}", font=("Courier",10), bg="#2e2e5c", fg="white").pack(anchor='w'); tk.Label(col_frame, text=f"  π:{p['pi1']:.4f}", font=("Courier",10), bg="#2e2e5c", fg="white").pack(anchor='w') 
                tk.Label(col_frame, text=" G2:", font=("Courier", 10, "bold"), bg="#2e2e5c", fg="magenta").pack(anchor='w', pady=(5,0)); tk.Label(col_frame, text=f"  μ:{p['mu2']:.4f}", font=("Courier",10), bg="#2e2e5c", fg="white").pack(anchor='w'); tk.Label(col_frame, text=f"  σ:{p['sigma2']:.4f}", font=("Courier",10), bg="#2e2e5c", fg="white").pack(anchor='w'); tk.Label(col_frame, text=f"  π:{p['pi2']:.4f}", font=("Courier",10), bg="#2e2e5c", fg="white").pack(anchor='w') 

    def show_3d_plot(self):
        if self.data is None: messagebox.showinfo("Información", "Simule primero."); return
        try:
            win3d = tk.Toplevel(self.window); win3d.title(f"3D - {self.dist_type}"); win3d.geometry("800x700")
            fig3d = plt.figure(figsize=(8, 7)); fig3d.patch.set_facecolor('#1a1a2e'); ax3d = fig3d.add_subplot(111, projection='3d'); ax3d.set_facecolor('#1a1a2e')
            title = ""
            if self.dist_type == "Normal Bivariada":
                params = {p: float(e.get()) for p, e in self.entries.items()}
                mu_x, mu_y, sigma_x, sigma_y, rho = (params["μ_x"], params["μ_y"], params["σ_x"], params["σ_y"], params["Correlación (ρ)"])
                x_range = [mu_x - 3.5*sigma_x, mu_x + 3.5*sigma_x]; y_range = [mu_y - 3.5*sigma_y, mu_y + 3.5*sigma_y]
                x_grid = np.linspace(x_range[0], x_range[1], 70); y_grid = np.linspace(y_range[0], y_range[1], 70); X, Y = np.meshgrid(x_grid, y_grid)
                pos = np.dstack((X, Y)); cov = [[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]; rv = stats.multivariate_normal([mu_x, mu_y], cov); Z = rv.pdf(pos)
                ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
                hist, xedges, yedges = np.histogram2d(self.data[:, 0], self.data[:, 1], bins=20, range=[x_range, y_range], density=True)
                xpos, ypos = np.meshgrid(xedges[:-1]+(xedges[1]-xedges[0])/2., yedges[:-1]+(yedges[1]-yedges[0])/2., indexing="ij"); xpos = xpos.ravel(); ypos = ypos.ravel(); zpos = 0
                dx = (xedges[1]-xedges[0])*np.ones_like(zpos); dy = (yedges[1]-yedges[0])*np.ones_like(zpos); dz = hist.ravel()
                ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='#9a7fdd', alpha=0.7)
                title = "Superficie y Muestra"
            elif self.dist_type == "Función Particular":
                x_grid = np.linspace(0, 2, 70); y_grid = np.linspace(0, 2, 70); X, Y = np.meshgrid(x_grid, y_grid); Z = (2*X + 3*Y + 2) / 28
                ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, edgecolor='none')
                ax3d.scatter(self.data[:, 0], self.data[:, 1], 0, c='#9a7fdd', alpha=0.4, s=10)
                ax3d.set_xlim(0, 2); ax3d.set_ylim(0, 2); title = "Superficie y Muestra"
            ax3d.set_xlabel('X'); ax3d.xaxis.label.set_color('white'); ax3d.set_ylabel('Y'); ax3d.yaxis.label.set_color('white'); ax3d.set_zlabel('Densidad'); ax3d.zaxis.label.set_color('white')
            ax3d.set_title(title, color="white"); ax3d.tick_params(axis='x', colors='white'); ax3d.tick_params(axis='y', colors='white'); ax3d.tick_params(axis='z', colors='white')
            canvas3d = FigureCanvasTkAgg(fig3d, master=win3d); canvas3d.get_tk_widget().pack(fill="both", expand=True); canvas3d.draw()
        except (ValueError, KeyError) as e: messagebox.showerror("Error", f"No se puede graficar 3D.\nError: {e}")
        except Exception as e: messagebox.showerror("Error", f"Error al crear gráfica 3D: {e}")

    def show_all_equations_window(self):
        if self.em_results is None: messagebox.showerror("Error", "Ejecute EM primero."); return
        eq_win = tk.Toplevel(self.window); eq_win.title("Ecuaciones Iteraciones"); eq_win.geometry("700x600"); eq_win.configure(bg="#1a1a2e")
        title_font = font.Font(family="Helvetica", size=16, weight="bold"); iter_font = font.Font(family="Helvetica", size=14, weight="bold", underline=False); eq_font = font.Font(family="Courier", size=14)
        tk.Label(eq_win, text="Ecuaciones de Mezcla de Normales por Iteración\n  π₁·N(x; μ₁, σ₁²) + π₂·N(y; μ₂, σ₂²)", font=title_font, bg="#1a1a2e", fg="white").pack(pady=(15, 10)) 
        main_frame = tk.Frame(eq_win, bg="#1a1a2e"); main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        canvas = tk.Canvas(main_frame, bg="#1a1a2e", highlightthickness=0); scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview); scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))); canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        for i, params in enumerate(self.em_results['steps']):
            iter_container = tk.Frame(scrollable_frame, bg="#2e2e5c", pady=10); iter_container.pack(fill="x", expand=True, pady=10, padx=5)
            tk.Label(iter_container, text=f"Iteración {i + 1}", font=iter_font, bg="#2e2e5c", fg="white").pack(pady=(0, 10))
            eq_frame = tk.Frame(iter_container, bg="#2e2e5c"); eq_frame.pack(pady=5, expand=True)
            pi1, mu1, sigma1 = params['pi1'], params['mu1'], params['sigma1']; pi2, mu2, sigma2 = params['pi2'], params['mu2'], params['sigma2']
            tk.Label(eq_frame, text=f" {pi1:.3f}", font=eq_font, bg="#2e2e5c", fg="cyan").pack(side="left"); tk.Label(eq_frame, text=f"·N(x; {mu1:.2f}, {sigma1**2:.2f})", font=eq_font, bg="#2e2e5c", fg="white").pack(side="left")
            tk.Label(eq_frame, text=" + ", font=eq_font, bg="#2e2e5c", fg="white").pack(side="left")
            tk.Label(eq_frame, text=f"{pi2:.3f}", font=eq_font, bg="#2e2e5c", fg="magenta").pack(side="left"); tk.Label(eq_frame, text=f"·N(y; {mu2:.2f}, {sigma2**2:.2f})", font=eq_font, bg="#2e2e5c", fg="white").pack(side="left")

# --- CLASE NUEVA: Ventana Bivariada ---
class BivariateEMWindow:
    def __init__(self, root, parent_window, df=None):
        self.root = root
        self.parent_window = parent_window
        self.df = df
        
        # --- Variables de Estado ---
        self.em_results = None       
        self.em_history = []         
        self.em_iteration_vars = []  
        self.show_final_mixture = False 
        self.show_final_params_var = tk.BooleanVar(value=True) 

        # Configuración Ventana
        self.window = tk.Toplevel(root)
        self.window.title("Algoritmo EM Bivariado - Avanzado")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_back)

        # Variables de Selección
        self.var_x = tk.StringVar(value="Seleccione X")
        self.var_y = tk.StringVar(value="Seleccione Y")
        self.num_iterations = tk.StringVar(value="10")

        # --- Layout Principal ---
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=10, pady=10)
        main_container.pack(fill="both", expand=True)

        # === PANEL IZQUIERDO ===
        left_panel = tk.Frame(main_container, bg="#2e2e5c", padx=8, pady=8, width=360)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False) 

        # 1. Título y Selección
        tk.Label(left_panel, text="EM Bivariado (2 Variables)", font=("Helvetica", 14, "bold"), bg="#2e2e5c", fg="white").pack(pady=(5, 10))
        
        sel_frame = tk.Frame(left_panel, bg="#2e2e5c")
        sel_frame.pack(fill="x", pady=2)
        
        tk.Label(sel_frame, text="Var X:", bg="#2e2e5c", fg="white", font=("Helvetica", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.menu_x = OptionMenu(sel_frame, self.var_x, "")
        self.menu_x.config(bg="#1a1a2e", fg="white", highlightthickness=0, width=22)
        self.menu_x.grid(row=0, column=1, padx=5, pady=2)

        tk.Label(sel_frame, text="Var Y:", bg="#2e2e5c", fg="white", font=("Helvetica", 9, "bold")).grid(row=1, column=0, sticky="w")
        self.menu_y = OptionMenu(sel_frame, self.var_y, "")
        self.menu_y.config(bg="#1a1a2e", fg="white", highlightthickness=0, width=22)
        self.menu_y.grid(row=1, column=1, padx=5, pady=2)

        # Iteraciones
        iter_frame = tk.Frame(left_panel, bg="#2e2e5c")
        iter_frame.pack(fill="x", pady=5)
        tk.Label(iter_frame, text="N° Iteraciones:", bg="#2e2e5c", fg="white").pack(side="left")
        tk.Entry(iter_frame, textvariable=self.num_iterations, width=5, bg="#1a1a2e", fg="white", insertbackground="white").pack(side="left", padx=10)

        # 2. Botones Principales
        btns_frame = tk.Frame(left_panel, bg="#2e2e5c")
        btns_frame.pack(fill="x", pady=5)

        self.btn_run = tk.Button(btns_frame, text="3. Ejecutar EM", command=self.run_bivariate_em, 
                  bg="#4CAF50", fg="white", font=("Helvetica", 9, "bold"), relief="flat")
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_save = tk.Button(btns_frame, text="4. Guardar Gráfica", command=self.save_plot, 
                  bg="#ff9800", fg="white", font=("Helvetica", 9, "bold"), relief="flat", state="disabled")
        self.btn_save.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # 3. Historial
        hist_frame = tk.Frame(left_panel, bg="#2e2e5c")
        hist_frame.pack(fill="x", pady=5)
        self.btn_history = tk.Button(hist_frame, text="Ver Historial", command=self.show_history_window, 
                  bg="#6c757d", fg="white", relief="flat", state="disabled", font=("Helvetica", 9))
        self.btn_history.pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(hist_frame, text="Limpiar Historial", command=self.clear_history, 
                  bg="#f44336", fg="white", relief="flat", font=("Helvetica", 9)).pack(side="right", fill="x", expand=True, padx=(2, 0))

        # 4. Mezcla Final y 3D
        mix_3d_frame = tk.Frame(left_panel, bg="#2e2e5c")
        mix_3d_frame.pack(fill="x", pady=(5, 2))

        self.btn_final_mix = tk.Button(mix_3d_frame, text="Mezcla Final", command=self.toggle_final_mixture, 
                  bg="#9a7fdd", fg="white", font=("Helvetica", 9, "bold"), relief="flat", state="disabled")
        self.btn_final_mix.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_3d = tk.Button(mix_3d_frame, text="Vista 3D", command=self.open_3d_window, 
                  bg="#00bcd4", fg="white", font=("Helvetica", 9, "bold"), relief="flat", state="disabled")
        self.btn_3d.pack(side="right", fill="x", expand=True, padx=(2, 0))

        self.chk_params = tk.Checkbutton(left_panel, text="Mostrar Parámetros Finales", variable=self.show_final_params_var, 
                       command=self.draw_plot, bg="#2e2e5c", fg="white", selectcolor="#1a1a2e", anchor="w", state="disabled")
        self.chk_params.pack(fill="x")

        self.btn_eq = tk.Button(left_panel, text="Ver Ecuaciones", command=self.show_equations_window, 
                  bg="#ff9800", fg="white", font=("Helvetica", 9, "bold"), relief="flat", state="disabled")
        self.btn_eq.pack(fill="x", pady=5)

        # BOTONES INFERIORES
        bottom_btns_frame = tk.Frame(left_panel, bg="#2e2e5c")
        bottom_btns_frame.pack(fill="x", pady=10, side="bottom")

        tk.Button(bottom_btns_frame, text="Limpiar", command=self.clear_all, 
                  bg="#f44336", fg="white", font=("Helvetica", 9, "bold"), relief="flat").pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        tk.Button(bottom_btns_frame, text="Regresar", command=self.go_back, 
                  bg="#6c757d", fg="white", font=("Helvetica", 9, "bold"), relief="flat").pack(side="right", fill="x", expand=True, padx=(2, 0))

        # 7. Superponer Iteraciones
        tk.Label(left_panel, text="Superponer Iteraciones:", font=("Helvetica", 10), bg="#2e2e5c", fg="white", anchor="w").pack(fill="x", pady=(5, 0))
        
        checklist_container = tk.Frame(left_panel, bg="#1a1a2e")
        checklist_container.pack(fill="both", expand=True, pady=5)
        
        self.canvas_chk = tk.Canvas(checklist_container, bg="#1a1a2e", highlightthickness=0)
        self.scrollbar_chk = tk.Scrollbar(checklist_container, orient="vertical", command=self.canvas_chk.yview)
        self.frame_chk = tk.Frame(self.canvas_chk, bg="#1a1a2e")

        # Configurar scroll
        self.frame_chk.bind("<Configure>", lambda e: self.canvas_chk.configure(scrollregion=self.canvas_chk.bbox("all")))
        self.canvas_chk.create_window((0, 0), window=self.frame_chk, anchor="nw")
        self.canvas_chk.configure(yscrollcommand=self.scrollbar_chk.set)

        self.canvas_chk.pack(side="left", fill="both", expand=True)
        self.scrollbar_chk.pack(side="right", fill="y")

        # Scroll Mouse
        def _on_mousewheel(event):
            self.canvas_chk.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas_chk.bind("<Enter>", lambda _: self.canvas_chk.bind_all("<MouseWheel>", _on_mousewheel))
        self.canvas_chk.bind("<Leave>", lambda _: self.canvas_chk.unbind_all("<MouseWheel>"))

        # === PANEL DERECHO (GRÁFICA) ===
        plot_frame = tk.Frame(main_container, bg="#1a1a2e")
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_facecolor('#2e2e5c')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        if self.df is not None:
            self.populate_menus()

    # --- MÉTODOS ---
    def go_back(self):
        self.window.destroy()
        self.parent_window.deiconify()

    def clear_all(self):
        self.em_results = None
        self.show_final_mixture = False
        self.ax.clear()
        self.ax.set_facecolor('#2e2e5c')
        self.canvas.draw()
        for widget in self.frame_chk.winfo_children(): widget.destroy()
        self.em_iteration_vars = []
        self.btn_save.config(state="disabled")
        self.btn_final_mix.config(state="disabled")
        self.btn_3d.config(state="disabled")
        self.chk_params.config(state="disabled")
        self.btn_eq.config(state="disabled")

    def clear_history(self):
        self.em_history = []
        self.btn_history.config(state="disabled")
        messagebox.showinfo("Historial", "Historial borrado.")

    def populate_menus(self):
        if self.df is None: return
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        menu_x = self.menu_x["menu"]; menu_x.delete(0, "end")
        menu_y = self.menu_y["menu"]; menu_y.delete(0, "end")
        for col in numeric_cols:
            menu_x.add_command(label=col, command=lambda v=col: self.var_x.set(v))
            menu_y.add_command(label=col, command=lambda v=col: self.var_y.set(v))
        if len(numeric_cols) >= 2:
            pc_cols = [c for c in numeric_cols if "PC" in c]
            if len(pc_cols) >= 2: self.var_x.set(pc_cols[0]); self.var_y.set(pc_cols[1])
            else: self.var_x.set(numeric_cols[0]); self.var_y.set(numeric_cols[1])

    def run_bivariate_em(self):
        col_x = self.var_x.get(); col_y = self.var_y.get()
        if self.df is None or col_x == "Seleccione X" or col_y == "Seleccione Y": return
        try: iters = int(self.num_iterations.get())
        except: return
        self.clear_all()

        data = self.df[[col_x, col_y]].dropna().values
        n_samples = len(data)
        
        # Inicialización
        data_sorted = data[data[:, 0].argsort()]
        mid = n_samples // 2
        mu1 = np.mean(data_sorted[:mid], axis=0); cov1 = np.cov(data_sorted[:mid], rowvar=False); pi1 = 0.5
        mu2 = np.mean(data_sorted[mid:], axis=0); cov2 = np.cov(data_sorted[mid:], rowvar=False); pi2 = 0.5
        reg_cov = 1e-6 * np.eye(2); cov1 += reg_cov; cov2 += reg_cov
        iteration_steps = []

        for i in range(iters):
            try:
                rv1 = multivariate_normal(mean=mu1, cov=cov1); rv2 = multivariate_normal(mean=mu2, cov=cov2)
                pdf1 = rv1.pdf(data); pdf2 = rv2.pdf(data)
                num1 = pi1 * pdf1; num2 = pi2 * pdf2
                denom = num1 + num2; denom[denom == 0] = 1e-10
                gamma1 = num1 / denom; gamma2 = num2 / denom
                
                N1 = np.sum(gamma1); N2 = np.sum(gamma2)
                pi1 = N1 / n_samples; pi2 = N2 / n_samples
                mu1 = np.sum(gamma1[:, np.newaxis] * data, axis=0) / N1
                mu2 = np.sum(gamma2[:, np.newaxis] * data, axis=0) / N2
                diff1 = data - mu1; diff2 = data - mu2
                cov1 = (gamma1[:, np.newaxis, np.newaxis] * diff1[:, :, np.newaxis] @ diff1[:, np.newaxis, :]).sum(axis=0) / N1
                cov2 = (gamma2[:, np.newaxis, np.newaxis] * diff2[:, :, np.newaxis] @ diff2[:, np.newaxis, :]).sum(axis=0) / N2
                cov1 += reg_cov; cov2 += reg_cov
                
                iteration_steps.append({"iter": i+1, "mu1": mu1, "cov1": cov1, "pi1": pi1, "mu2": mu2, "cov2": cov2, "pi2": pi2})
            except: break

        self.em_results = {"steps": iteration_steps, "data": data, "cols": (col_x, col_y)}
        self.em_history.append({"vars": f"{col_x} vs {col_y}", "steps": iteration_steps})
        
        self.btn_save.config(state="normal"); self.btn_final_mix.config(state="normal")
        self.btn_3d.config(state="normal"); self.chk_params.config(state="normal")
        self.btn_eq.config(state="normal"); self.btn_history.config(state="normal")

        for i in range(len(iteration_steps)):
            iter_frame = tk.Frame(self.frame_chk, bg="#1a1a2e")
            iter_frame.pack(fill="x", pady=2)
            tk.Label(iter_frame, text=f"Iter {i+1}:", bg="#1a1a2e", fg="white", width=6).pack(side="left")
            var1 = tk.BooleanVar(); cb1 = tk.Checkbutton(iter_frame, text="N1", variable=var1, command=self.draw_plot, bg="#1a1a2e", fg="cyan", selectcolor="#2e2e5c"); cb1.pack(side="left")
            var2 = tk.BooleanVar(); cb2 = tk.Checkbutton(iter_frame, text="N2", variable=var2, command=self.draw_plot, bg="#1a1a2e", fg="magenta", selectcolor="#2e2e5c"); cb2.pack(side="left")
            varM = tk.BooleanVar(); cbM = tk.Checkbutton(iter_frame, text="Mezcla", variable=varM, command=self.draw_plot, bg="#1a1a2e", fg="yellow", selectcolor="#2e2e5c"); cbM.pack(side="left")
            self.em_iteration_vars.append((var1, var2, varM))

        self.draw_plot()

    def toggle_final_mixture(self):
        self.show_final_mixture = not self.show_final_mixture
        self.draw_plot()

    def draw_plot(self):
        if self.em_results is None: return
        data = self.em_results["data"]; steps = self.em_results["steps"]; col_x, col_y = self.em_results["cols"]
        
        self.ax.clear(); self.ax.set_facecolor('#2e2e5c')
        self.ax.scatter(data[:, 0], data[:, 1], c='#9a7fdd', alpha=0.3, s=15, label="Datos")
        
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        pos = np.dstack((xx, yy))

        for i, (v1, v2, vM) in enumerate(self.em_iteration_vars):
            if v1.get() or v2.get() or vM.get():
                step = steps[i]
                rv1 = multivariate_normal(step['mu1'], step['cov1']); z1 = rv1.pdf(pos) * step['pi1']
                rv2 = multivariate_normal(step['mu2'], step['cov2']); z2 = rv2.pdf(pos) * step['pi2']
                if v1.get(): self.ax.contour(xx, yy, z1, colors='cyan', linestyles='dashed', linewidths=1, alpha=0.7)
                if v2.get(): self.ax.contour(xx, yy, z2, colors='magenta', linestyles='dashed', linewidths=1, alpha=0.7)
                if vM.get(): self.ax.contour(xx, yy, z1 + z2, colors='yellow', linestyles='dashed', linewidths=1, alpha=0.7)

        if self.show_final_mixture and steps:
            final = steps[-1]
            rv1f = multivariate_normal(final['mu1'], final['cov1']); z1f = rv1f.pdf(pos) * final['pi1']
            rv2f = multivariate_normal(final['mu2'], final['cov2']); z2f = rv2f.pdf(pos) * final['pi2']
            self.ax.contour(xx, yy, z1f, colors='cyan', linewidths=2, alpha=0.9)
            self.ax.contour(xx, yy, z2f, colors='magenta', linewidths=2, alpha=0.9)
            self.ax.contour(xx, yy, z1f + z2f, colors='gold', linewidths=2, alpha=1.0)
            
            if self.show_final_params_var.get():
                s1 = np.sqrt(np.diag(final['cov1'])); s2 = np.sqrt(np.diag(final['cov2']))
                txt1 = f"N1 (Cian)\nπ={final['pi1']:.3f}\nμ=({final['mu1'][0]:.3f}, {final['mu1'][1]:.3f})\nσ=({s1[0]:.3f}, {s1[1]:.3f})"
                self.ax.text(final['mu1'][0], final['mu1'][1], txt1, color="cyan", fontsize=8, bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'))
                txt2 = f"N2 (Magenta)\nπ={final['pi2']:.3f}\nμ=({final['mu2'][0]:.3f}, {final['mu2'][1]:32f})\nσ=({s2[0]:.3f}, {s2[1]:.3f})"
                self.ax.text(final['mu2'][0], final['mu2'][1], txt2, color="magenta", fontsize=8, bbox=dict(facecolor='black', alpha=0.7, edgecolor='magenta'))

        self.ax.set_title(f"Análisis EM: {col_x} vs {col_y}", color="white"); self.ax.tick_params(colors='white')
        self.ax.set_xlabel(col_x, color="white"); self.ax.set_ylabel(col_y, color="white")
        self.canvas.draw()

    def open_3d_window(self):
        if self.em_results is None: return
        self.window.withdraw()
        BivariateEM3DWindow(self.root, self.window, self.em_results)

    def save_plot(self):
        if self.em_results is None: return
        try:
            filename = f"EM_2D_{self.em_results['cols'][0]}_{self.em_results['cols'][1]}.png"
            os.makedirs("em_gallery", exist_ok=True)
            self.fig.savefig(f"em_gallery/{filename}", facecolor=self.fig.get_facecolor(), dpi=150)
            messagebox.showinfo("Guardado", f"Guardado en em_gallery/{filename}")
        except Exception as e: messagebox.showerror("Error", str(e))

    # --- AQUÍ ESTÁ LA MAGIA: Ventana de Ecuaciones con Scroll ---
    def show_equations_window(self):
        if self.em_results is None:
            messagebox.showwarning("Aviso", "Ejecute el algoritmo primero.")
            return

        win = tk.Toplevel(self.window)
        win.title("Ecuaciones por Iteración")
        win.geometry("900x600")
        win.configure(bg="#1a1a2e")
        
        # Título
        tk.Label(win, text="Evolución de la Mezcla de Normales Bivariadas", 
                 font=("Helvetica", 16, "bold"), bg="#1a1a2e", fg="white").pack(pady=(20, 10))
        
        tk.Label(win, text="f(x) = π₁·N(x | μ₁, Σ₁) + π₂·N(x | μ₂, Σ₂)", 
                 font=("Consolas", 14), bg="#1a1a2e", fg="#d1c4e9").pack(pady=(0, 20))

        # --- Área con Scroll ---
        container = tk.Frame(win, bg="#1a1a2e")
        container.pack(fill="both", expand=True, padx=20, pady=10)
        
        canvas = tk.Canvas(container, bg="#1a1a2e", highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=840) # Width fijo para centrar
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        win.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>")) 

        # Generar Tarjetas (Estilo Imagen)
        steps = self.em_results["steps"]
        for i, step in enumerate(steps):
            # Frame de la tarjeta
            card = tk.Frame(scrollable_frame, bg="#2e2e5c", pady=10, padx=10)
            card.pack(fill="x", pady=5, padx=5)
            
            # Título Iteración
            tk.Label(card, text=f"Iteración {i+1}", font=("Helvetica", 12, "bold"), bg="#2e2e5c", fg="white").pack()
            
            # Valores
            pi1, mu1 = step['pi1'], step['mu1']
            pi2, mu2 = step['pi2'], step['mu2']
            
            # Función helper para imprimir vectores
            def vec_str(v): return f"[{v[0]:.2f}, {v[1]:.2f}]"
            
            # Ecuación visual
            eq_frame = tk.Frame(card, bg="#2e2e5c")
            eq_frame.pack(pady=5)
            
            # Parte 1 (Cian)
            tk.Label(eq_frame, text=f"{pi1:.3f}", fg="cyan", bg="#2e2e5c", font=("Consolas", 11, "bold")).pack(side="left")
            tk.Label(eq_frame, text=f"·N(x; {vec_str(mu1)}, Σ₁)  ", fg="#d1c4e9", bg="#2e2e5c", font=("Consolas", 11)).pack(side="left")
            
            # Más (+)
            tk.Label(eq_frame, text="+  ", fg="white", bg="#2e2e5c", font=("Consolas", 11)).pack(side="left")
            
            # Parte 2 (Magenta)
            tk.Label(eq_frame, text=f"{pi2:.3f}", fg="magenta", bg="#2e2e5c", font=("Consolas", 11, "bold")).pack(side="left")
            tk.Label(eq_frame, text=f"·N(x; {vec_str(mu2)}, Σ₂)", fg="#d1c4e9", bg="#2e2e5c", font=("Consolas", 11)).pack(side="left")

    def show_history_window(self):
        win = tk.Toplevel(self.window); win.title("Historial"); win.geometry("600x500"); win.configure(bg="#1a1a2e")
        txt = scrolledtext.ScrolledText(win, bg="#1a1a2e", fg="#00ff00"); txt.pack(fill="both", expand=True)
        for i, h in enumerate(self.em_history): txt.insert(tk.END, f"Ejecución {i+1}: {h['vars']} ({len(h['steps'])} pasos)\n")

# --- NUEVA CLASE: VENTANA 3D ---
class BivariateEM3DWindow:
    def __init__(self, root, parent_window, em_results):
        self.root = root
        self.parent_window = parent_window
        self.em_results = em_results
        
        self.window = tk.Toplevel(root)
        self.window.title("Vista 3D - Algoritmo EM")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_back)

        # Variables para checklist
        self.iteration_vars_3d = []

        # Layout
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=10, pady=10)
        main_container.pack(fill="both", expand=True)

        # Panel Izquierdo (Checklist y Regresar)
        left_panel = tk.Frame(main_container, bg="#2e2e5c", padx=10, pady=10, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="Vista 3D Iteraciones", font=("Helvetica", 16, "bold"), bg="#2e2e5c", fg="white").pack(pady=(5, 20))

        # --- Checklist Scrollable MEJORADO ---
        chk_container = tk.Frame(left_panel, bg="#1a1a2e")
        chk_container.pack(fill="both", expand=True, pady=(0, 20))
        
        self.canvas = tk.Canvas(chk_container, bg="#1a1a2e", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(chk_container, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg="#1a1a2e")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind("<Enter>", lambda _: self.canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.canvas.bind("<Leave>", lambda _: self.canvas.unbind_all("<MouseWheel>"))

        # Generar checklist
        steps = self.em_results["steps"]
        for i in range(len(steps)):
            row = tk.Frame(self.scrollable_frame, bg="#1a1a2e")
            row.pack(fill="x", pady=2, padx=5)
            
            lbl = tk.Label(row, text=f"Iter {i+1}:", fg="white", bg="#1a1a2e", width=6)
            lbl.pack(side="left")
            
            var = tk.BooleanVar()
            chk = tk.Checkbutton(row, text="Ver Superficie", variable=var, command=self.draw_3d, bg="#1a1a2e", fg="cyan", selectcolor="#2e2e5c", anchor="w")
            chk.pack(side="left", fill="x", expand=True)
            self.iteration_vars_3d.append(var)

        tk.Button(left_panel, text="Regresar a Bivariada", command=self.go_back, 
                  bg="#dc3545", fg="white", font=("Helvetica", 10, "bold"), relief="flat").pack(side="bottom", fill="x")

        # Panel Derecho (Gráfica 3D)
        plot_frame = tk.Frame(main_container, bg="#1a1a2e")
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1a1a2e')
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        
        self.draw_3d()

    def go_back(self):
        self.window.destroy()
        self.parent_window.deiconify()

    def draw_3d(self):
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        
        data = self.em_results["data"]
        col_x, col_y = self.em_results["cols"]
        self.ax.scatter(data[:, 0], data[:, 1], 0, c='#9a7fdd', alpha=0.2, s=10)
        
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
        pos = np.dstack((xx, yy))

        steps = self.em_results["steps"]
        
        for i, var in enumerate(self.iteration_vars_3d):
            if var.get():
                step = steps[i]
                rv1 = multivariate_normal(step['mu1'], step['cov1'])
                rv2 = multivariate_normal(step['mu2'], step['cov2'])
                z = (rv1.pdf(pos) * step['pi1']) + (rv2.pdf(pos) * step['pi2'])
                self.ax.plot_surface(xx, yy, z, cmap='viridis', alpha=0.5, rstride=2, cstride=2, edgecolor='none')

        self.ax.set_xlabel(col_x, color="white"); self.ax.set_ylabel(col_y, color="white"); self.ax.set_zlabel("Densidad", color="white")
        self.ax.tick_params(colors='white')
        self.canvas_plot.draw()

# --- CLASE NUEVA: Clasificador Bivariado ---
class BivariateDiagnosisWindow:
    def __init__(self, root, main_menu):
        self.root = root
        self.main_menu = main_menu
        self.window = tk.Toplevel(root)
        self.window.title("Clasificador Bivariado (Diagnóstico)")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg="#1a1a2e")
        self.window.protocol("WM_DELETE_WINDOW", self.go_back)
        
        self.df = None
        self.loaded_base_filename = ""
        self.var_x = tk.StringVar(value="Seleccionar")
        self.var_y = tk.StringVar(value="Seleccionar")

        # --- Layout Principal ---
        main_container = tk.Frame(self.window, bg="#1a1a2e", padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        # Título
        tk.Label(main_container, text="Clasificador Bivariado (2 Variables Independientes)", 
                 font=("Helvetica", 22, "bold"), bg="#1a1a2e", fg="white").pack(pady=(0, 20))

        # --- SECCIÓN 1: PARÁMETROS (Pre-cargados) ---
        param_frame = tk.LabelFrame(main_container, text="Configuración de Clases (Normal Bivariada)", 
                                    bg="#2e2e5c", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=10)
        param_frame.pack(fill="x", pady=(0, 20))

        # Encabezados
        headers_frame = tk.Frame(param_frame, bg="#2e2e5c")
        headers_frame.pack(fill="x")
        tk.Label(headers_frame, text="Clase", width=15, bg="#2e2e5c", fg="white", font=("Helvetica", 10, "bold")).pack(side="left")
        labels = ["Peso (π)", "Media X (μ1)", "Media Y (μ2)", "Desv.Std X (σ1)", "Desv.Std Y (σ2)"]
        for l in labels:
            tk.Label(headers_frame, text=l, width=15, bg="#2e2e5c", fg="#d1c4e9").pack(side="left", padx=2)

        # Función para crear fila de entradas
        def create_row(parent, name, color, defaults):
            row = tk.Frame(parent, bg="#2e2e5c")
            row.pack(fill="x", pady=5)
            tk.Label(row, text=name, width=15, bg="#2e2e5c", fg=color, font=("Helvetica", 11, "bold"), anchor="w").pack(side="left")
            entries = []
            for val in defaults:
                e = tk.Entry(row, width=15, bg="#1a1a2e", fg="white", insertbackground="white", justify="center")
                e.insert(0, str(val))
                e.pack(side="left", padx=2)
                entries.append(e)
            return entries

        # >>> TUS VALORES AQUÍ <<<
        # N1: Benigno (Cian) - pi=0.55, mu=(-1.43, -0.16), sigma=(0.79, 0.56)
        self.ents_n1 = create_row(param_frame, "N1 (Benigno)", "cyan", [0.55, -1.43, -0.16, 0.79, 0.56])
        
        # N2: Maligno (Lila) - pi=0.45, mu=(1.75, 0.19), sigma=(2.16, 1.25)
        self.ents_n2 = create_row(param_frame, "N2 (Maligno)", "magenta", [0.45, 1.75, 0.19, 2.16, 1.25])

        # --- SECCIÓN 2: CARGA DE DATOS Y SELECCIÓN ---
        ctrl_frame = tk.Frame(main_container, bg="#1a1a2e")
        ctrl_frame.pack(fill="x", pady=(0, 10))

        btn_style = {"font": ("Helvetica", 11, "bold"), "relief": "flat", "width": 18}

        tk.Button(ctrl_frame, text="1. Cargar Archivo", command=self.load_file, bg="#007bff", fg="white", **btn_style).pack(side="left", padx=(0, 10))

        # Selectores de variables
        tk.Label(ctrl_frame, text="Variable X:", bg="#1a1a2e", fg="white").pack(side="left")
        self.menu_x = OptionMenu(ctrl_frame, self.var_x, "")
        self.menu_x.config(bg="#2e2e5c", fg="white", width=12, highlightthickness=0)
        self.menu_x.pack(side="left", padx=(5, 15))

        tk.Label(ctrl_frame, text="Variable Y:", bg="#1a1a2e", fg="white").pack(side="left")
        self.menu_y = OptionMenu(sel_frame := tk.Frame(ctrl_frame, bg="#1a1a2e"), self.var_y, "") 
        self.menu_y = OptionMenu(ctrl_frame, self.var_y, "")
        self.menu_y.config(bg="#2e2e5c", fg="white", width=12, highlightthickness=0)
        self.menu_y.pack(side="left", padx=5)

        self.btn_run = tk.Button(ctrl_frame, text="2. Localizar", command=self.run_classification, bg="#28a745", fg="white", state="disabled", **btn_style)
        self.btn_run.pack(side="left", padx=20)

        self.btn_download = tk.Button(ctrl_frame, text="Descargar Excel", command=self.download_excel, bg="#ff9800", fg="white", state="disabled", **btn_style)
        self.btn_download.pack(side="left", padx=5)

        tk.Button(ctrl_frame, text="Regresar", command=self.go_back, bg="#6c757d", fg="white", **btn_style).pack(side="right")

        # --- SECCIÓN 3: RESULTADOS VISUALES ---
        split_frame = tk.Frame(main_container, bg="#1a1a2e")
        split_frame.pack(fill="both", expand=True)

        # 1. Tabla Izquierda: Ubicación Cartesiana
        left_result = tk.LabelFrame(split_frame, text="Ubicación Cartesiana (Valores X, Y y Densidad)", bg="#1a1a2e", fg="white", font=("Helvetica", 11))
        left_result.pack(side="left", fill="both", expand=True, padx=(0, 5))

        cols_left = ("Indice", "Valor X", "Valor Y", "Densidad (Max)")
        self.tree_left = ttk.Treeview(left_result, columns=cols_left, show='headings')
        for c in cols_left:
            self.tree_left.heading(c, text=c)
            self.tree_left.column(c, width=70, anchor="center")
        
        vsb_left = ttk.Scrollbar(left_result, orient="vertical", command=self.tree_left.yview)
        self.tree_left.configure(yscrollcommand=vsb_left.set)
        self.tree_left.pack(side="left", fill="both", expand=True)
        vsb_left.pack(side="right", fill="y")

        # 2. Centro: Gráfica (Pequeña)
        plot_frame = tk.LabelFrame(split_frame, text="Visualización", bg="#1a1a2e", fg="white")
        plot_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        self.fig = plt.Figure(figsize=(3, 3), dpi=100)
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2e2e5c')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # 3. Tabla Derecha: Diagnóstico Final
        right_result = tk.LabelFrame(split_frame, text="Diagnóstico Final", bg="#1a1a2e", fg="white", font=("Helvetica", 11))
        right_result.pack(side="right", fill="both", expand=True, padx=(5, 0))

        cols_right = ("Indice", "Valor X", "Valor Y", "Diagnóstico", "Clase")
        self.tree_right = ttk.Treeview(right_result, columns=cols_right, show='headings')
        for c in cols_right:
            self.tree_right.heading(c, text=c)
            self.tree_right.column(c, width=70, anchor="center")
        
        vsb_right = ttk.Scrollbar(right_result, orient="vertical", command=self.tree_right.yview)
        self.tree_right.configure(yscrollcommand=vsb_right.set)
        self.tree_right.pack(side="left", fill="both", expand=True)
        vsb_right.pack(side="right", fill="y")

        # Tags de color
        self.tree_right.tag_configure("benign", foreground="cyan")
        self.tree_right.tag_configure("malignant", foreground="magenta")

    def go_back(self):
        self.window.destroy()
        self.main_menu.deiconify()

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=(("Excel/CSV", "*.xlsx *.csv"), ("All files", "*.*")))
        if not filepath: return
        try:
            self.loaded_base_filename = os.path.splitext(os.path.basename(filepath))[0]
            if filepath.endswith('.csv'): self.df = pd.read_csv(filepath)
            else: self.df = pd.read_excel(filepath)
            
            # Llenar los menús
            nums = self.df.select_dtypes(include=np.number).columns.tolist()
            m_x = self.menu_x["menu"]; m_x.delete(0, "end")
            m_y = self.menu_y["menu"]; m_y.delete(0, "end")
            
            for c in nums:
                m_x.add_command(label=c, command=lambda v=c: self.var_x.set(v))
                m_y.add_command(label=c, command=lambda v=c: self.var_y.set(v))
            
            # Auto-seleccionar si hay columnas PC (PC1, PC2)
            pc_cols = [c for c in nums if "PC" in c or "Component" in c]
            if len(pc_cols) >= 2:
                self.var_x.set(pc_cols[0])
                self.var_y.set(pc_cols[1])
            elif len(nums) >= 2:
                self.var_x.set(nums[0])
                self.var_y.set(nums[1])
                
            self.btn_run.config(state="normal")
            self.btn_download.config(state="disabled")
            messagebox.showinfo("Carga Exitosa", f"Archivo cargado: {len(self.df)} filas")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_classification(self):
        col_x = self.var_x.get()
        col_y = self.var_y.get()
        
        if col_x == "Seleccionar" or col_y == "Seleccionar" or col_x == col_y:
            messagebox.showwarning("Error", "Seleccione dos variables distintas (X y Y).")
            return

        try:
            X_val = self.df[col_x].values
            Y_val = self.df[col_y].values
            
            # --- Obtener Parámetros de los inputs ---
            # N1 (Benigno)
            pi1 = float(self.ents_n1[0].get())
            mu1 = [float(self.ents_n1[1].get()), float(self.ents_n1[2].get())]
            # Asumimos covarianza diagonal (independencia) si solo se da sigma
            cov1 = [[float(self.ents_n1[3].get())**2, 0], [0, float(self.ents_n1[4].get())**2]]
            
            # N2 (Maligno)
            pi2 = float(self.ents_n2[0].get())
            mu2 = [float(self.ents_n2[1].get()), float(self.ents_n2[2].get())]
            cov2 = [[float(self.ents_n2[3].get())**2, 0], [0, float(self.ents_n2[4].get())**2]]

            # Crear modelos
            rv1 = multivariate_normal(mu1, cov1)
            rv2 = multivariate_normal(mu2, cov2)

            # Limpiar GUI
            for item in self.tree_left.get_children(): self.tree_left.delete(item)
            for item in self.tree_right.get_children(): self.tree_right.delete(item)
            self.ax.clear()

            colors = []
            # Clasificación Punto a Punto
            for i in range(len(X_val)):
                x, y = X_val[i], Y_val[i]
                # Probabilidad (Densidad * Peso)
                prob1 = rv1.pdf([x, y]) * pi1
                prob2 = rv2.pdf([x, y]) * pi2
                
                max_dens = max(prob1, prob2) # Usamos la probabilidad ponderada como medida de densidad relativa
                
                if prob2 > prob1: # Si Maligno es mayor
                    diag = "M"
                    clase = "Maligno (Lila)"
                    tag = "malignant"
                    c = "magenta"
                else: # Si Benigno es mayor o igual
                    diag = "B"
                    clase = "Benigno (Cian)"
                    tag = "benign"
                    c = "cyan"
                
                # Llenar Tablas
                self.tree_left.insert("", "end", values=(i+1, round(x,3), round(y,3), f"{max_dens:.4f}"))
                self.tree_right.insert("", "end", values=(i+1, round(x,3), round(y,3), diag, clase), tags=(tag,))
                colors.append(c)

            # Graficar Dispersión
            self.ax.scatter(X_val, Y_val, c=colors, s=20, edgecolors='white', linewidth=0.5, alpha=0.7)
            
            # Graficar Contornos
            x_min, x_max = X_val.min()-1, X_val.max()+1
            y_min, y_max = Y_val.min()-1, Y_val.max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            pos = np.dstack((xx, yy))
            
            self.ax.contour(xx, yy, rv1.pdf(pos), colors='cyan', alpha=0.5, levels=4)
            self.ax.contour(xx, yy, rv2.pdf(pos), colors='magenta', alpha=0.5, levels=4)

            self.ax.set_xlabel(col_x, color="white", fontsize=8)
            self.ax.set_ylabel(col_y, color="white", fontsize=8)
            self.ax.set_title("Clasificación Bivariada", color="white", fontsize=10)
            self.ax.tick_params(colors="white", labelsize=7)
            self.canvas.draw()
            
            self.btn_download.config(state="normal")
            messagebox.showinfo("Completado", "Clasificación finalizada correctamente.")

        except Exception as e:
            messagebox.showerror("Error Numérico", f"Revise que los datos y parámetros sean numéricos.\n{e}")

    def download_excel(self):
        if not self.tree_right.get_children(): return
        try:
            # Recopilar Datos Tabla Derecha (Diagnóstico)
            data_diag = []
            for item in self.tree_right.get_children():
                data_diag.append(self.tree_right.item(item)['values'])
            df_diag = pd.DataFrame(data_diag, columns=["Indice", "Valor X", "Valor Y", "Diagnóstico", "Clase"])

            # Recopilar Datos Tabla Izquierda (Ubicación)
            data_loc = []
            for item in self.tree_left.get_children():
                data_loc.append(self.tree_left.item(item)['values'])
            df_loc = pd.DataFrame(data_loc, columns=["Indice", "Valor X", "Valor Y", "Densidad (Max)"])

            # Generar Nombre Archivo
            def clean(s): return "".join(c for c in s.replace(" ", "_") if c.isalnum() or c in ('_', '-'))
            
            base_name = f"{clean(self.loaded_base_filename)}_Bivariado"
            filename = f"{base_name}.xlsx"
            save_path = os.path.join(CARPETA_DIAGNOSTICO, filename)
            
            count = 1
            while os.path.exists(save_path):
                save_path = os.path.join(CARPETA_DIAGNOSTICO, f"{base_name}_V{count}.xlsx")
                count += 1

            # Guardar Excel
            with pd.ExcelWriter(save_path) as writer:
                df_diag.to_excel(writer, sheet_name="Diagnostico Final", index=False)
                df_loc.to_excel(writer, sheet_name="Ubicación Cartesiana", index=False)
            
            messagebox.showinfo("Guardado", f"Archivo guardado en:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error al guardar", f"No se pudo guardar el archivo:\n{e}")
            
if __name__ == "__main__":
    main_root = tk.Tk()
    app = App(main_root)
    main_root.mainloop()