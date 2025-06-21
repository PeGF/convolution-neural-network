# descriptors.py

import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import pywt
import cv2


# ==========================
# HOG - Histogram of Oriented Gradients
# ==========================
def extract_hog(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extrai o descritor HOG de um conjunto de imagens.
    
    Args:
        images (numpy array): Array de imagens no formato (N, altura, largura).
        pixels_per_cell (tuple): Tamanho da célula.
        cells_per_block (tuple): Tamanho do bloco.
        orientations (int): Número de orientações.

    Returns:
        features (numpy array): Array com os descritores HOG extraídos.
    """
    hog_features = []
    for img in images:
        feature = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False
        )
        hog_features.append(feature)
    return np.array(hog_features)


# ==========================
# LBP - Local Binary Pattern
# ==========================
def extract_lbp(images, P=8, R=1):
    """
    Extrai o descritor LBP de um conjunto de imagens.

    Args:
        images (numpy array): Array de imagens no formato (N, altura, largura).
        P (int): Número de pontos vizinhos.
        R (float): Raio.

    Returns:
        features (numpy array): Array flatten com histogramas de LBP.
    """
    lbp_features = []
    for img in images:
        lbp = local_binary_pattern(img, P, R, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, P + 3),
                                 range=(0, P + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalização
        lbp_features.append(hist)
    return np.array(lbp_features)


# ==========================
# Wavelet Haar
# ==========================
def extract_haar(images):
    """
    Aplica transformada wavelet Haar nas imagens.

    Args:
        images (numpy array): Array de imagens (N, altura, largura).

    Returns:
        features (numpy array): Coeficientes de aproximação flatten.
    """
    haar_features = []
    for img in images:
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        haar_features.append(cA.flatten())  # Usamos apenas a aproximação
    return np.array(haar_features)
