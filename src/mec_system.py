import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, impulse, step, residue
from scipy.signal import tf2zpk


class MechanicalSystem:
    def __init__(self, mass: float, damping: float, stiffness: float) -> None:
        """
        Inicializa um sistema mecânico de segunda ordem.
        
        Args:
            mass: Massa do sistema [kg]
            damping: Coeficiente de amortecimento [N·s/m]
            stiffness: Constante de rigidez [N/m]
        """
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        
        # Coeficientes da função de transferência
        self._numerator = [1.0]
        self._denominator = [mass, damping, stiffness]  
        self.transfer_function = TransferFunction(self._numerator, self._denominator)
        self._zeros, self._poles, _ = tf2zpk(self._numerator, self._denominator)

    def plot_poles_zeros(self) -> None:
        """Plota o diagrama de polos e zeros do sistema."""
        plt.figure(figsize=(6, 6))
        
        # Configurações do gráfico
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plot dos polos
        plt.scatter(
            np.real(self._poles), 
            np.imag(self._poles), 
            marker='x', 
            color='red', 
            s=100, 
            label='Polos'
        )
        
        # Plot dos zeros (se existirem)
        if len(self._zeros) > 0:
            plt.scatter(
                np.real(self._zeros), 
                np.imag(self._zeros), 
                marker='o', 
                color='blue', 
                s=100, 
                label='Zeros'
            )
            
        plt.title('Mapa de Polos e Zeros')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def partial_fraction_expansion(self) -> tuple:
        """
        Realiza a expansão em frações parciais da função de transferência.
        
        Returns:
            tuple: (resíduos, polos, coeficientes)
        """
        residues, poles, coefficients = residue(self._numerator, self._denominator)
        return residues, poles, coefficients

    def plot_partial_fraction_components(self) -> None:
        """Plota os componentes da expansão em frações parciais."""
        residues, poles, _ = self.partial_fraction_expansion()
        time = np.linspace(0, 20, 500)
        total_response = np.zeros_like(time)

        plt.figure(figsize=(8, 6))
        
        # Plota cada componente da expansão
        for i, (residue, pole) in enumerate(zip(residues, poles)):
            component = np.real(residue * np.exp(pole * time))
            total_response += component
            plt.plot(time, component, label=f'Componente {i+1}')

        # Plota a soma total
        plt.plot(time, total_response, 'k--', label='Soma Total')
        
        # Configurações do gráfico
        plt.title('Componentes da Transformada Inversa de Laplace')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_impulse_response(self) -> None:
        """Plota a resposta ao impulso do sistema."""
        time, response = impulse(self.transfer_function)
        
        plt.figure(figsize=(6, 4))
        plt.plot(time, response)
        plt.title('Resposta ao Impulso')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_step_response(self) -> None:
        """Plota a resposta ao degrau do sistema."""
        time, response = step(self.transfer_function)
        
        plt.figure(figsize=(6, 4))
        plt.plot(time, response)
        plt.title('Resposta ao Degrau')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Parâmetros do sistema
    MASS = 1.0        # Massa [kg]
    DAMPING = 0.1     # Coeficiente de amortecimento [N·s/m]
    STIFFNESS = 0.5   # Constante de rigidez [N/m]
    
    system = MechanicalSystem(MASS, DAMPING, STIFFNESS)
    system.plot_poles_zeros()
    system.plot_partial_fraction_components()
    system.plot_impulse_response()
    system.plot_step_response()