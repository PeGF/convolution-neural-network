from src.train import train_task
from src.evaluate import load_training_log, plot_training_curves, evaluate_model
from src.config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Treinamento
    history = train_task(mode="Multiclass")

    # Avaliação
    history = load_training_log(mode="Multiclass")
    plot_training_curves(history, mode="Multiclass")
    evaluate_model(mode="Multiclass")


if __name__ == "__main__":
    main()
