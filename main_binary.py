from src.data_loader import load_mnist_data, load_data_with_descriptors
from src.model_builder import build_cnn_model, create_mlp
from src.train import train_task
from src.evaluate import load_training_log, plot_training_curves, evaluate_model
from src.config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Treinamento
    history_bin = train_task(mode="binary")

    # Avaliação
    history_bin = load_training_log('binary')
    plot_training_curves(history_bin, 'binary')
    # class_a e class_b são carregados do config ou passados aqui
    evaluate_model('binary', *BINARY_CLASSES)


if __name__ == "__main__":
    main()
