import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, impulse, step, residue

class SistemaEletrico:
    def __init__(self):
        # Parâmetros do circuito
        self.L1 = 10e-3        # 10 mH
        self.L2 = 15e-3        # 15 mH
        self.R1 = 100          # 100 Ohms
        self.R2 = 150          # 150 Ohms
        self.C = 100e-9        # 100 nF
        
        # Coeficientes da função de transferência
        self.coef_s2 = self.C * (self.L1 + self.L2)
        self.coef_s1 = self.C * (self.R1 + self.R2)
        self.coef_s0 = 1 
        
        # Criar sistema
        self.numerador = [1]
        self.denominador = [self.coef_s2, self.coef_s1, self.coef_s0]
        self.sistema = TransferFunction(self.numerador, self.denominador)
        
        # Calcular polos e zeros
        self.polos = np.roots(self.denominador)
        self.zeros = np.roots(self.numerador) if len(self.numerador) > 1 else []
        
        # Calcular frações parciais
        self.residuos, self.polos_parciais, _ = residue(self.numerador, self.denominador)
    
    def plot_polos_zeros(self):
        """Plota o mapa de polos e zeros do sistema."""
        plt.figure()
        plt.scatter(np.real(self.polos), np.imag(self.polos), marker='x', color='red', label='Polos')
        if len(self.zeros) > 0:
            plt.scatter(np.real(self.zeros), np.imag(self.zeros), marker='o', color='blue', label='Zeros')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("Mapa de Polos e Zeros - Sistema Elétrico")
        plt.xlabel("Parte Real")
        plt.ylabel("Parte Imaginária")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_resposta_impulso(self):
        """Plota a resposta ao impulso do sistema."""
        tempo, resposta = impulse(self.sistema)
        plt.figure()
        plt.plot(tempo, resposta)
        plt.title("Resposta ao Impulso - Sistema Elétrico")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    
    def plot_resposta_degrau(self):
        """Plota a resposta ao degrau do sistema."""
        tempo, resposta = step(self.sistema)
        plt.figure()
        plt.plot(tempo, resposta)
        plt.title("Resposta ao Degrau - Sistema Elétrico")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    
    def plot_fracao_parcial(self, tempo_max=0.002):
        """Plota as componentes das frações parciais."""
        tempo = np.linspace(0, tempo_max, 1000)
        soma_componentes = np.zeros_like(tempo)
        
        plt.figure()
        for i in range(len(self.residuos)):
            componente = np.real(self.residuos[i] * np.exp(self.polos_parciais[i] * tempo))
            soma_componentes += componente
            plt.plot(tempo, componente, label=f'Componente {i+1}')
        
        plt.plot(tempo, soma_componentes, 'k--', label='Soma das Componentes')
        plt.title("Componentes da Fração Parcial - Sistema Elétrico")
        plt.xlabel("Tempo [s]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    sistema = SistemaEletrico()
    sistema.plot_polos_zeros()
    sistema.plot_resposta_impulso()
    sistema.plot_resposta_degrau()
    sistema.plot_fracao_parcial()