import torch
import platform

print(f"Machine : {platform.machine()}")
print(f"Système : {platform.system()}")
print(f"PyTorch Version : {torch.__version__}")

# Vérification du GPU Apple Silicon (MPS - Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\n✅ SUCCÈS : L'accélération GPU (Metal) est activée !")
    print("Ton Mac M2 va faire tourner les réseaux de neurones à toute vitesse.")

    # Petit test de calcul sur le GPU
    x = torch.ones(1, device=device)
    print(f"Test tenseur créé sur : {x.device}")

else:
    print("\n❌ ATTENTION : Le mode MPS n'est pas détecté.")
    print("Le code va tourner sur le CPU (plus lent).")
