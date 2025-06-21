from data_loader import load_data, load_data_with_descriptors
from src.model_builder import create_cnn, create_mlp
from train import train
from evaluate import evaluate
from config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🔸 Escolha se quer usar descritores ou imagens cruas
    USE_DESCRIPTORS = False  # 👉 Altere para True se quiser usar HOG, LBP ou Haar
    DESCRIPTOR_TYPE = 'hog'  # 'hog', 'lbp' ou 'haar'

    if USE_DESCRIPTORS:
        X_train, X_test, y_train, y_test = load_data_with_descriptors(descriptor=DESCRIPTOR_TYPE)
        input_shape = X_train.shape[1]
        model = create_mlp(input_shape=input_shape, num_classes=NUM_CLASSES)
    else:
        X_train, X_test, y_train, y_test = load_data()
        input_shape = X_train.shape[1:]  # (28, 28, 1)
        model = create_cnn(input_shape=input_shape, num_classes=NUM_CLASSES)

    # 🔧 Treinamento
    history = train(model, X_train, y_train)

    # 📊 Avaliação
    evaluate(model, history, X_test, y_test, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
