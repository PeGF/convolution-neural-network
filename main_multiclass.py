from src.data_loader import load_mnist_data, load_data_with_descriptors
from src.model_builder import build_cnn_model, create_mlp
from src.train import train_task
from src.evaluate import load_training_log, plot_training_curves, evaluate_model
from src.config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🔸 Escolha se quer usar descritores ou imagens cruas
    USE_DESCRIPTORS = False  # 👉 Altere para True se quiser usar HOG, LBP ou Haar
    DESCRIPTOR_TYPE = 'hog'  # 'hog', 'lbp' ou 'haar'

    if USE_DESCRIPTORS:
        (x_train, y_train), (x_test, y_test) = load_data_with_descriptors(descriptor=DESCRIPTOR_TYPE)
        input_shape = x_train.shape[1]
        model = create_mlp(input_shape=input_shape, num_classes=NUM_CLASSES)
    else:
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        input_shape = x_train.shape[1:]  # (28, 28, 1)
        model = build_cnn_model(num_classes=NUM_CLASSES)

    # Treinamento
    history = train_task(mode="multiclass")

    # Avaliação
    history = load_training_log(mode="multiclass")
    plot_training_curves(history, mode="multiclass")
    evaluate_model(mode="multiclass")


if __name__ == "__main__":
    main()
