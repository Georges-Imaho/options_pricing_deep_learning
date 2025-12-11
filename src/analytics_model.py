import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    """
    Implémentation vectorisée du modèle Black-Scholes Merton (1973).
    Sert d'Oracle (Label Generator) pour l'entraînement des réseaux de neurones.
    """

    def __init__(self, r=0.0, sigma=0.2):
        """
        :param r: Taux sans risque (Risk-free rate), ex: 0.05 pour 5%
        :param sigma: Volatilité de l'actif sous-jacent (ex: 0.2 pour 20%)
        """
        self.r = r
        self.sigma = sigma

    def calculate_price(self, S, K, T, option_type='call'):
        """
        Calcule le prix de l'option Européenne (Call ou Put).
        Tous les arguments peuvent être des scalaires ou des tableaux NumPy (Vectors).

        :param S: Prix actuel de l'actif (Spot Price)
        :param K: Prix d'exercice (Strike Price)
        :param T: Temps restant avant maturité (en années)
        :param option_type: 'call' ou 'put'
        :return: Prix de l'option (même dimension que S)
        """
        # Sécurité pour éviter la division par zéro si T=0 (expiration)
        # On remplace les 0 par une valeur infime
        T = np.maximum(T, 1e-8)

        # Calcul des termes d1 et d2
        # d1 mesure la sensibilité du prix par rapport au sous-jacent
        # d2 mesure la probabilité que l'option finisse dans la monnaie
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)

        if option_type == 'call':
            # Formule du Call : S * N(d1) - K * exp(-rT) * N(d2)
            price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        elif option_type == 'put':
            # Formule du Put : K * exp(-rT) * N(-d2) - S * N(-d1)
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

        return price

    def calculate_delta(self, S, K, T, option_type='call'):
        """
        Calcule le Delta (La dérivée du prix par rapport à S).
        Essentiel pour la partie Hedging (Phase 3).
        """
        T = np.maximum(T, 1e-8)
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))

        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

# --- ZONE DE TEST (Pour vérifier que tout marche) ---
if __name__ == "__main__":
    # Paramètres de test
    S0 = 100    # Prix actuel
    K = 100     # Strike (ATM - At The Money)
    T = 1.0     # 1 an
    r = 0.05    # 5% taux
    sigma = 0.2 # 20% vol

    bs = BlackScholesModel(r=r, sigma=sigma)

    call_price = bs.calculate_price(S0, K, T, 'call')
    put_price = bs.calculate_price(S0, K, T, 'put')

    print(f"--- Test Unitaire Black-Scholes ---")
    print(f"Spot: {S0}, Strike: {K}, T: {T}, Vol: {sigma}")
    print(f"Call Price théorique : {call_price:.4f}") # Devrait être autour de 10.45
    print(f"Put Price théorique  : {put_price:.4f}")  # Devrait être autour de 5.57 (Call - S + K*exp(-rT))

    # Test Vectorisé (Simulation de 5 options d'un coup)
    S_vector = np.array([80, 90, 100, 110, 120])
    prices_vector = bs.calculate_price(S_vector, K, T, 'call')
    print(f"\nPricing Vectorisé (5 spots différents) : \n{prices_vector}")