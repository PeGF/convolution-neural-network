# src/data_loader.py

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from descriptors import extract_hog, extract_lbp, extract_haar
from keras.datasets import mnist
from src.config import BINARY_CLASSES, RANDOM_SEED


def load_mnist_data():
    """
    Carrega o dataset MNIST diretamente do TensorFlow.
    Retorna os dados de treino e teste, ainda sem pré-processamento.
    """
    (x_train, y_train), (x_test, y_test) = mnist.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocess_images(images):
    """
    Normaliza as imagens para valores entre 0 e 1
    e adiciona o canal (grayscale), convertendo de (28, 28) para (28, 28, 1).

    Parâmetros:
    - images: numpy array de imagens (shape: [n amostras, 28, 28])

    Retorna:
    - imagens normalizadas (shape: [n amostras, 28, 28, 1])
    """
    images = images.astype('float32') / 255.0  # Normalização para [0, 1]
    images = np.expand_dims(images, -1)        # Adiciona o canal (grayscale)
    return images


def prepare_multiclass_data():
    """
    Prepara os dados para a tarefa de classificação multiclasse (0 a 9).

    Retorna:
    - x_train, x_test: imagens normalizadas
    - y_train, y_test: labels (inteiros de 0 a 9)
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Pré-processamento das imagens
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    return (x_train, y_train), (x_test, y_test)


def prepare_binary_data(class_a=BINARY_CLASSES[0], class_b=BINARY_CLASSES[1]):
    """
    Prepara os dados para a tarefa de classificação binária entre duas classes específicas.

    Parâmetros:
    - class_a: primeira classe (ex.: 3)
    - class_b: segunda classe (ex.: 8)

    Retorna:
    - x_train, x_test: imagens normalizadas
    - y_train, y_test: labels (0 para class_a, 1 para class_b)
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Filtra as amostras das classes escolhidas
    train_filter = np.isin(y_train, [class_a, class_b])
    test_filter = np.isin(y_test, [class_a, class_b])

    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # Mapeia as labels:
    # class_a -> 0
    # class_b -> 1
    y_train = np.where(y_train == class_a, 0, 1)
    y_test = np.where(y_test == class_a, 0, 1)

    # Pré-processamento das imagens
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    return (x_train, y_train), (x_test, y_test)

def load_data_with_descriptors(descriptor='hog', binary_classes=None):
    '''
    Carrega o dataset MNIST e aplica o descritor escolhido.

    Args:
        descriptor (str): 'hog', 'lbp' ou 'haar'
        binary_classes (tuple): Exemplo (0, 1) se for classificação binária

    Returns:
        X_train, X_test, y_train, y_test (arrays numpy)
    '''

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if binary_classes is not None:
        idx_train = np.isin(y_train, binary_classes)
        idx_test = np.isin(y_test, binary_classes)
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]
        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

        # Ajusta os labels para 0 e 1
        y_train = (y_train == binary_classes[1]).astype(int)
        y_test = (y_test == binary_classes[1]).astype(int)

    # Normalização para 0-1 (necessário para alguns descritores)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Aplicação do descritor escolhido
    if descriptor == "hog":
        x_train_feat = extract_hog(x_train)
        x_test_feat = extract_hog(x_test)
    elif descriptor == "lbp":
        x_train_feat = extract_lbp(x_train)
        x_test_feat = extract_lbp(x_test)
    elif descriptor == "haar":
        x_train_feat = extract_haar(x_train)
        x_test_feat = extract_haar(x_test)
    else:
        raise ValueError("Descritor não reconhecido. Use 'hog', 'lbp' ou 'haar'.")

    return (x_train_feat, y_train), (x_test_feat, y_test)

def split_train_validation(x, y, validation_size=0.1):
    """
    Divide os dados de treino em treino e validação.

    Parâmetros:
    - x: imagens de treino
    - y: labels de treino
    - validation_size: proporção para validação (padrão 10%)

    Retorna:
    - x_train, x_val, y_train, y_val
    """
    return train_test_split(
        x, y,
        test_size=validation_size,
        random_state=RANDOM_SEED,
        stratify=y  # Mantém a proporção das classes
    )
