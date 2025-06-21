from data_loader import load_data, load_data_with_descriptors
from src.model_builder import create_cnn, create_mlp
from train import train
from evaluate import evaluate
from config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🔸 Defina quais classes usar para binário
    BINARY_CLASSES = (0, 1)  # 👉 Pode trocar (ex.: (3, 8))

    # 🔸 Escolha se quer usar descritores ou imagens cruas
    USE_DESCRIPTORS = False  # 👉 Altere para True se quiser usar HOG, LBP ou Haar
    DESCRIPTOR_TYPE = 'hog'  # 'hog', 'lbp' ou 'haar'

    if USE_DESCRIPTORS:
        X_train, X_test, y_train, y_test = load_data_with_descriptors(
            descriptor=DESCRIPTOR_TYPE,
            binary_classes=BINARY_CLASSES
        )
        input_shape = X_train.shape[1]
        model = create_mlp(input_shape=input_shape, num_classes=1)
    else:
        X_train, X_test, y_train, y_test = load_data(binary_classes=BINARY_CLASSES)
        input_shape = X_train.shape[1:]  # (28, 28, 1)
        model = create_cnn(input_shape=input_shape, num_classes=1)

    # 🔧 Treinamento
    history = train(model, X_train, y_train)

    # 📊 Avaliação
    evaluate(model, history, X_test, y_test, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
