from src.data_loader import load_mnist_data, load_data_with_descriptors
from src.model_builder import build_cnn_model, create_mlp
from src.train import train_task
from src.evaluate import load_training_log, plot_training_curves, evaluate_model
from src.config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Treinamento
    history = train_task(mode="multiclass")

    # Avaliação
    history = load_training_log(mode="multiclass")
    plot_training_curves(history, mode="multiclass")
    evaluate_model(mode="multiclass")


if __name__ == "__main__":
    main()
