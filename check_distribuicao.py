import os
import numpy as np

from src.data_loader import load_mnist_data
from src.config import BINARY_CLASSES

# Carrega os dados
(x_train, y_train), (x_test, y_test) = load_mnist_data()


# Binário
print("\n-DISTRIBUIÇÃO BINÁRIA-")

# Filtra para as classes binárias do config
mask_train = np.isin(y_train, BINARY_CLASSES)
y_train_bin = y_train[mask_train]

mask_test = np.isin(y_test, BINARY_CLASSES)
y_test_bin = y_test[mask_test]

# Teino
classes, counts = np.unique(y_train_bin, return_counts=True)
print("Treino (binário):")
for c, n in zip(classes, counts):
    print(f"  Classe {c}: {n} exemplos")

# Teste
classes, counts = np.unique(y_test_bin, return_counts=True)
print("Teste (binário):")
for c, n in zip(classes, counts):
    print(f"  Classe {c}: {n} exemplos")


# Multiclasse
print("\n-DISTRIBUIÇÃO MULTICLASSE-")

# Treino
classes, counts = np.unique(y_train, return_counts=True)
print("Treino (multiclasse):")
for c, n in zip(classes, counts):
    print(f"  Classe {c}: {n} exemplos")

# Teste
classes, counts = np.unique(y_test, return_counts=True)
print("Teste (multiclasse):")
for c, n in zip(classes, counts):
    print(f"  Classe {c}: {n} exemplos")
