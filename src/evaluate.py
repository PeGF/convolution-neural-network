import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

sns.set_theme(style="whitegrid")

from src.config import OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS
from src.data_loader import (
    prepare_multiclass_data,
    prepare_binary_data,
    split_train_validation
)


def load_training_log(mode="multiclass"):
    """
    Carrega o arquivo CSV do histórico de treinamento.

    Parâmetros:
    - mode: 'multiclass' ou 'binary'

    Retorna:
    - DataFrame do pandas com colunas loss, accuracy, val_loss, val_accuracy
    """
    path = os.path.join(OUTPUT_DIR, f"{mode}_training_log.csv")
    return pd.read_csv(path)


def plot_training_curves(history_df, mode="multiclass"):
    """
    Gera e salva gráficos de loss e accuracy ao longo das épocas.

    Parâmetros:
    - history_df: DataFrame resultante de load_training_log()
    - mode: 'multiclass' ou 'binary'
    """
    # Curva de perda (loss)
    colors = {'train': '#1f77b4', 'val': '#ff7f0e'}  # Azul e Laranja

    plt.figure(figsize=(8, 6))
    plt.plot(history_df['loss'], label='Treino', color=colors['train'], linewidth=2)
    plt.plot(history_df['val_loss'], label='Validação', color=colors['val'], linewidth=2, linestyle='--')
    plt.title(f'Curva de Loss por Época ({mode})', fontsize=16)
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mode}_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df['accuracy'], label='Treino', color=colors['train'], linewidth=2)
    plt.plot(history_df['val_accuracy'], label='Validação', color=colors['val'], linewidth=2, linestyle='--')
    plt.title(f'Curva de Accuracy por Época ({mode})', fontsize=16)
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mode}_accuracy_curve.png"), dpi=300)
    plt.close()

def evaluate_model(mode="multiclass", class_a=None, class_b=None):
    """
    Avalia o modelo com base nas saídas salvas e gera métricas e matriz de confusão.

    Parâmetros:
    - mode: 'multiclass' ou 'binary'
    - class_a, class_b: utilizados apenas em modo binary para indicar as classes originais
    """
    # Carregar dados de teste originais e saídas da rede
    if mode == 'multiclass':
        (_, _), (x_test, y_test) = prepare_multiclass_data()
        outputs = np.load(os.path.join(OUTPUT_DIR, f"{mode}_test_outputs.npz"))['outputs']
        # Para multiclasse, outputs têm shape (n, 10) de probabilidades
        y_pred = np.argmax(outputs, axis=1)
    else:
        (_, _), (x_test, y_test_binary) = prepare_binary_data(class_a, class_b)
        outputs = np.load(os.path.join(OUTPUT_DIR, f"{mode}_test_outputs.npz"))['outputs']
        # Para binário, outputs têm shape (n, 1), threshold em 0.5
        y_pred = (outputs > 0.5).astype(int).squeeze()
        y_test = y_test_binary

    # Cálculo de acurácia geral
    acc = accuracy_score(y_test, y_pred)

    # Geração de matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Geração de relatório de classificação
    report = classification_report(y_test, y_pred)

    # Salvar métricas em arquivos
    with open(os.path.join(OUTPUT_DIR, f"{mode}_accuracy.txt"), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    with open(os.path.join(OUTPUT_DIR, f"{mode}_classification_report.txt"), 'w') as f:
        f.write(report)

    # Plotar e salvar matriz de confusão
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray')
    plt.title(f'Matriz de Confusão ({mode})', fontsize=16)
    plt.xlabel('Predito', fontsize=14)
    plt.ylabel('Verdadeiro', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mode}_confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"Avaliação ({mode}) concluída. Acurácia: {acc:.4f}")


if __name__ == '__main__':

    # Exemplo multiclasse
    history_mult = load_training_log('multiclass')
    plot_training_curves(history_mult, 'multiclass')
    evaluate_model('multiclass')

    # Exemplo binário
    history_bin = load_training_log('binary')
    plot_training_curves(history_bin, 'binary')
    # class_a e class_b são carregados do config ou passados aqui
    from src.config import BINARY_CLASSES
    evaluate_model('binary', *BINARY_CLASSES)
