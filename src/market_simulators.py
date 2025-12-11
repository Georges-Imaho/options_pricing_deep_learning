import numpy as np
import matplotlib.pyplot as plt


class MarketSimulator:
    def __init__(self, S0=100, drift=0.05, vol=0.2, T=1.0, dt=1/252):
        self.S0 = S0
        self.drift = drift
        self.vol = vol
        self.T = T
        self.dt = dt
        self.N_steps =int(T/dt)

    def simulate(self, n_paths=1000):
        Z = np.random.normal(0, 1, size=(n_paths, self.N_steps))

        drift_term = (self.drift - 0.5*self.vol**2)*self.dt

        diffusion_term = self.vol * Z * np.sqrt(self.dt)

        log_returns = diffusion_term + drift_term

        log_returns = np.hstack([np.zeros((n_paths, 1)), log_returns])

        cumulative_log_returns = np.cumsum(log_returns, axis=1)

        paths = self.S0 * np.exp(cumulative_log_returns)

        return paths


if __name__ == "__main__":
    # Paramètres de simulation
    simulator = MarketSimulator(S0=100, drift=0.05, vol=0.2, T=1.0)

    # Génération de 10 trajectoires pour visualiser
    paths = simulator.simulate(n_paths=50)

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(paths.T)  # .T pour transposer (temps en axe X)
    plt.title("Simulation de Monte Carlo (Black-Scholes / GBM)")
    plt.xlabel("Jours de trading")
    plt.ylabel("Prix de l'actif ($)")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Dimension du dataset généré : {paths.shape}")
    print("Dataset prêt pour l'entraînement !")