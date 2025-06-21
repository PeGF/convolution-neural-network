from src.data_loader import load_mnist_data, load_data_with_descriptors
from src.model_builder import build_cnn_model, create_mlp
from src.train import train_task
from src.evaluate import load_training_log, plot_training_curves, evaluate_model
from src.config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    USE_DESCRIPTORS = False  # Alterar para True se quiser usar HOG, LBP ou Haar
    DESCRIPTOR_TYPE = 'hog'  # 'hog', 'lbp' ou 'haar'

    if USE_DESCRIPTORS:
        X_train, X_test, y_train, y_test = load_data_with_descriptors(
            descriptor=DESCRIPTOR_TYPE,
            binary_classes=BINARY_CLASSES
        )
        input_shape = X_train.shape[1]
        model = create_mlp(input_shape=input_shape, num_classes=1)
    else:
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
        input_shape = x_train.shape[1:]  # (28, 28, 1)
        model = build_cnn_model(NUM_CLASSES)

    # Treinamento
    history_bin = train_task(mode="binary")

    # Avaliação
    history_bin = load_training_log('binary')
    plot_training_curves(history_bin, 'binary')
    # class_a e class_b são carregados do config ou passados aqui
    evaluate_model('binary', *BINARY_CLASSES)


if __name__ == "__main__":
    main()
