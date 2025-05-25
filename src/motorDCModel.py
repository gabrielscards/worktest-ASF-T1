import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal

class MotorAnalysis:
    def __init__(self):
        # Parâmetros do motor
        self.Ld = 0.02  # H
        self.Rd = 0.5   # Ω
        self.Kf = 5     # N·m/A
        self.Ke = 2     # V·s/rad
        self.Jm = 5     # kg·m²
        self.b = 2      # N·s/m
        
        # Função de transferência
        self.num = [self.Kf]
        self.den = [self.Ld*self.Jm, self.Ld*self.b + self.Rd*self.Jm, self.Rd*self.b + self.Kf*self.Ke]
        self.sys = ctrl.TransferFunction(self.num, self.den)
        
        # Polos e zeros
        self.poles = np.roots(self.den)
        self.zeros = np.roots(self.num)
        
    def plot_pole_zero(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(np.real(self.poles), np.imag(self.poles), marker='x', color='red', label='Polos')
        plt.scatter(np.real(self.zeros), np.imag(self.zeros), marker='o', color='blue', label='Zeros')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.title('Mapa de Polos e Zeros')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def partial_fraction_expansion(self):
        # Expansão em frações parciais
        r, p, k = signal.residue(self.num, self.den)
        
        # Criar funções de transferência para cada termo
        t = np.linspace(0, 2, 1000)
        components = []
        
        for residue, pole in zip(r, p):
            sys = ctrl.TransferFunction([residue], [1, -pole])
            t, y = ctrl.impulse_response(sys, t)
            components.append((t, y, f'Resíduo: {residue:.2f}, Polo: {pole:.2f}'))
            
        plt.figure(figsize=(10, 6))
        for t, y, label in components:
            plt.plot(t, y, label=label)
        plt.title('Componentes da Expansão em Frações Parciais')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_impulse_response(self):
        t = np.linspace(0, 2, 1000)
        t, y = ctrl.impulse_response(self.sys, t)
        
        plt.figure(figsize=(8, 6))
        plt.plot(t, y)
        plt.title('Resposta ao Impulso (Delta de Dirac)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade Angular (rad/s)')
        plt.grid(True)
        plt.show()
        
    def plot_step_response(self):
        t = np.linspace(0, 2, 1000)
        t, y = ctrl.step_response(self.sys, t)
        
        plt.figure(figsize=(8, 6))
        plt.plot(t, y)
        plt.title('Resposta ao Degrau Unitário')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade Angular (rad/s)')
        plt.grid(True)
        plt.show()
        
    def run_all_analysis(self):
        print("Função de Transferência:")
        print(f"H(s) = {self.Kf} / ({self.den[0]:.3f}s² + {self.den[1]:.3f}s + {self.den[2]:.3f})")
        print("\nPolos:", self.poles)
        print("Zeros:", self.zeros)
        
        self.plot_pole_zero()
        self.partial_fraction_expansion()
        self.plot_impulse_response()
        self.plot_step_response()

# Executar análise
motor = MotorAnalysis()
motor.run_all_analysis()