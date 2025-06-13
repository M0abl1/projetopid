# projetopid

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim
from scipy.linalg import eigvals

import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# --- 1. Definição da Função de Transferência do Processo P(s) ---
# P(s) = 6 / (s^2 + 8.3s + 13.42)
num_P = [6]
den_P = [1, 8.3, 13.42]
P_s = TransferFunction(num_P, den_P)

# --- 2. Função para o Design do Controlador PID via Alocação de Polos ---
def design_pid_pole_placement(desired_poles):
    """
    Calcula os ganhos Kp, Ki, Kd de um controlador PID
    usando o método de alocação de polos.
    """
    coeffs_desired_char_eq = np.poly(desired_poles)
    alpha2 = coeffs_desired_char_eq[1]
    alpha1 = coeffs_desired_char_eq[2]
    alpha0 = coeffs_desired_char_eq[3]

    Kd = (alpha2 - 8.3) / 6
    Kp = (alpha1 - 13.42) / 6
    Ki = alpha0 / 6

    return Kp, Ki, Kd

# --- Função de Simulação e Plotagem (Adaptada para GUI) ---
def simulate_and_plot_gui(s3_estimated, step_amplitude, ax):
    """
    Simula a resposta do sistema de malha fechada e atualiza o gráfico no Axes fornecido.
    """
    # Limpa o Axes antes de plotar novos dados
    ax.clear()

    # Definir polos desejados (p1 e p2 como par complexo conjugado dominante)
    zeta = 0.707
    omega_n = 2.0
    p1 = -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta**2)
    p2 = -zeta * omega_n - 1j * omega_n * np.sqrt(1 - zeta**2)

    desired_poles_total = [p1, p2, s3_estimated]

    # Calcular os ganhos PID
    Kp, Ki, Kd = design_pid_pole_placement(desired_poles_total)

    # Impressões para o console (ainda úteis para debug e memorial)
    print(f"\n--- Simulação com S3={s3_estimated}, a={step_amplitude} ---")
    print(f"Ganhos PID: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}")

    # Função de Transferência do Controlador PID: C(s) = (Kd*s^2 + Kp*s + Ki) / s
    num_C = [Kd, Kp, Ki]
    den_C = [1, 0]
    C_s = TransferFunction(num_C, den_C)

    # Função de Transferência de Malha Aberta (L(s) = C(s) * P(s))
    num_L = np.polymul(C_s.num, P_s.num)
    den_L = np.polymul(C_s.den, P_s.den)
    L_s = TransferFunction(num_L, den_L)

    # Função de Transferência de Malha Fechada (T(s) = L(s) / (1 + L(s)))
    num_T = num_L
    den_T = np.polyadd(den_L, num_L)
    T_s = TransferFunction(num_T, den_T)

    closed_loop_poles = np.roots(T_s.den)
    print("Polos de Malha Fechada:", closed_loop_poles)

    # Simulação da Resposta ao Degrau
    time_end = 15
    num_points = 1000
    t = np.linspace(0, time_end, num_points)
    r_t = step_amplitude * np.ones_like(t)

    tout, y_out, _ = lsim(T_s, r_t, t) # Desempacota o terceiro valor (estados) para evitar ValueError

    # Plotagem
    ax.plot(tout, y_out, label='Saída do Sistema y(t)', color='blue')
    ax.plot(tout, r_t, linestyle='--', color='red', label=f'Entrada de Referência R(t) = {step_amplitude}')
    ax.set_title(f'Resposta do Sistema com PID (S3_estimado={s3_estimated})')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Redesenha o canvas para mostrar as atualizações
    canvas.draw()

# --- Configuração da Interface Gráfica ---
def setup_gui():
    root = tk.Tk()
    root.title("Controlador PID - Simulação")
    root.geometry("800x700") # Tamanho inicial da janela

    # Frame para as entradas de parâmetros
    input_frame = tk.Frame(root, padx=10, pady=10)
    input_frame.pack(side=tk.TOP, fill=tk.X)

    tk.Label(input_frame, text="Polo S3 Estimado (negativo, ex: -5):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    s3_entry = tk.Entry(input_frame)
    s3_entry.insert(0, "-5.0") # Valor padrão
    s3_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    tk.Label(input_frame, text="Amplitude da Entrada Degrau 'a':").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    amplitude_entry = tk.Entry(input_frame)
    amplitude_entry.insert(0, "1.0") # Valor padrão
    amplitude_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    # Frame para os resultados dos ganhos (opcional, apenas para exibição)
    results_frame = tk.Frame(root, padx=10, pady=5, relief=tk.GROOVE, bd=2)
    results_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    tk.Label(results_frame, text="Ganhos PID:").pack(anchor="w")
    kp_label = tk.Label(results_frame, text="Kp: ")
    kp_label.pack(anchor="w")
    ki_label = tk.Label(results_frame, text="Ki: ")
    ki_label.pack(anchor="w")
    kd_label = tk.Label(results_frame, text="Kd: ")
    kd_label.pack(anchor="w")


    # Função para o botão de simulação
    def on_simulate():
        try:
            s3 = float(s3_entry.get())
            amplitude = float(amplitude_entry.get())

            if s3 >= 0:
                messagebox.showerror("Erro de Entrada", "O polo S3 deve ser um valor negativo (estável).")
                return

            # Calcular e exibir os ganhos (fora da função de plotagem para poder exibi-los)
            zeta = 0.707
            omega_n = 2.0
            p1 = -zeta * omega_n + 1j * omega_n * np.sqrt(1 - zeta**2)
            p2 = -zeta * omega_n - 1j * omega_n * np.sqrt(1 - zeta**2)
            desired_poles_calc = [p1, p2, s3]
            Kp_val, Ki_val, Kd_val = design_pid_pole_placement(desired_poles_calc)

            kp_label.config(text=f"Kp: {Kp_val:.4f}")
            ki_label.config(text=f"Ki: {Ki_val:.4f}")
            kd_label.config(text=f"Kd: {Kd_val:.4f}")

            # Chamar a função de simulação e plotagem
            simulate_and_plot_gui(s3, amplitude, ax)

        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro na simulação: {e}")

    simulate_button = tk.Button(input_frame, text="Simular Resposta", command=on_simulate)
    simulate_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Configuração do gráfico Matplotlib
    fig = Figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111)

    global canvas # Torna canvas global para ser acessível pela função simulate_and_plot_gui
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Adicionar barra de ferramentas do Matplotlib (zoom, pan, etc.)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Executar uma simulação inicial ao iniciar a GUI
    # on_simulate() # Descomente se quiser que um gráfico seja exibido ao iniciar

    root.mainloop()

if __name__ == "__main__":
    setup_gui()
