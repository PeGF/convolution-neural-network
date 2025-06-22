from src.train import train_task
from src.evaluate import load_training_log, plot_training_curves, evaluate_model
from src.config import *
import os


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Treinamento
    history_bin = train_task(mode="Binary")

    # Avaliação
    history_bin = load_training_log('Binary')
    plot_training_curves(history_bin, 'Binary')
    # class_a e class_b são carregados do config ou passados aqui
    evaluate_model('Binary', *BINARY_CLASSES)


if __name__ == "__main__":
    main()
