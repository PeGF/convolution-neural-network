import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

sns.set_theme(style="whitegrid")

from src.config import OUTPUT_DIR, BATCH_SIZE, NUM_EPOCAS, CLASS_NAMES
from src.data_loader import (
    prepare_multiclass_data,
    prepare_binary_data,
    split_train_validation
)


def load_training_log(mode):
    """
    Carrega o arquivo CSV do histórico de treinamento.

    Parâmetros:
    - mode: 'multiclass' ou 'binary'

    Retorna:
    - DataFrame do pandas com colunas loss, accuracy, val_loss, val_accuracy
    """
    path = os.path.join(OUTPUT_DIR, f"{mode}_training_log.csv")
    return pd.read_csv(path)


def plot_training_curves(history_df, mode):
    from src.config import NUM_EPOCAS

    colors = {'train': '#1f77b4', 'val': '#ff7f0e'}

    x = np.arange(1, len(history_df) + 1)  # Agora de 1 até NUM_EPOCAS

    plt.figure(figsize=(12, 10))
    plt.plot(x, history_df['loss'], label='Treino', color=colors['train'], linewidth=2)
    plt.plot(x, history_df['val_loss'], label='Validação', color=colors['val'], linewidth=2, linestyle='--')

    last_epoch = x[-1]
    plt.text(last_epoch + 0.01, history_df['loss'].iloc[-1], f"{history_df['loss'].iloc[-1]:.4f}",
             color='black', fontsize=13, fontweight='bold', va='center', ha='left', fontfamily='serif')
    plt.text(last_epoch + 0.01, history_df['val_loss'].iloc[-1], f"{history_df['val_loss'].iloc[-1]:.4f}",
             color='black', fontsize=13, fontweight='bold', va='center', ha='left', fontfamily='serif')

    plt.title(f'Curva de Loss por Época ({mode})', fontsize=16, fontweight='bold', pad=18, fontfamily='serif')
    plt.xlabel('Época', fontsize=14, fontweight='bold', fontfamily='serif')
    plt.ylabel('Loss', fontsize=14, fontweight='bold', fontfamily='serif')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(
        ticks=np.arange(1, NUM_EPOCAS + 1, max(1, NUM_EPOCAS // 10)),
        labels=[str(i) for i in np.arange(1, NUM_EPOCAS + 1, max(1, NUM_EPOCAS // 10))],
        fontsize=12,
        fontfamily='serif'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mode}_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.plot(x, history_df['accuracy'], label='Treino', color=colors['train'], linewidth=2)
    plt.plot(x, history_df['val_accuracy'], label='Validação', color=colors['val'], linewidth=2, linestyle='--')

    plt.text(last_epoch + 0.01, history_df['accuracy'].iloc[-1], f"{history_df['accuracy'].iloc[-1]:.4f}",
             color='black', fontsize=13, fontweight='bold', va='center', ha='left', fontfamily='serif')
    plt.text(last_epoch + 0.01, history_df['val_accuracy'].iloc[-1], f"{history_df['val_accuracy'].iloc[-1]:.4f}",
             color='black', fontsize=13, fontweight='bold', va='center', ha='left', fontfamily='serif')

    plt.title(f'Curva de Accuracy por Época ({mode})', fontsize=16, fontweight='bold', pad=18, fontfamily='serif')
    plt.xlabel('Época', fontsize=14, fontweight='bold', fontfamily='serif')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold', fontfamily='serif')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(
        ticks=np.arange(1, NUM_EPOCAS + 1, max(1, NUM_EPOCAS // 10)),
        labels=[str(i) for i in np.arange(1, NUM_EPOCAS + 1, max(1, NUM_EPOCAS // 10))],
        fontsize=12,
        fontfamily='serif'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mode}_accuracy_curve.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_labels, mode):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens', linewidths=1.5, linecolor='gray',
                xticklabels=class_labels, yticklabels=class_labels, square=True, cbar=False, ax=ax)

    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = (count / row_sums[i][0] * 100) if row_sums[i][0] > 0 else 0
            ax.text(j + 0.5, i + 0.35, f"{count}", ha='center', va='center', fontsize=18, color='black', fontfamily='serif')
            ax.text(j + 0.5, i + 0.65, f"{percent:.1f}%", ha='center', va='center', fontsize=11, color='black', fontfamily='serif')

    plt.title(f'Matriz de Confusão ({mode})', fontsize=18, fontweight='bold', pad=22, fontfamily='serif')
    plt.xlabel('Predito', fontsize=16, fontweight='bold', fontfamily='serif')
    plt.ylabel('Verdadeiro', fontsize=16, fontweight='bold', fontfamily='serif')
    plt.xticks(rotation=45, fontsize=12, fontfamily='serif')
    plt.yticks(rotation=0, fontsize=12, fontfamily='serif')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{mode}_confusion_matrix.png"), dpi=300)
    plt.close()

def evaluate_model(mode, class_a=None, class_b=None):
    """
    Avalia o modelo com base nas saídas salvas e gera métricas e matriz de confusão.
    """
    # Carrega dados de teste originais e saídas da rede
    if mode == 'Multiclass':
        (_, _), (x_test, y_test) = prepare_multiclass_data()
        outputs = np.load(os.path.join(OUTPUT_DIR, f"{mode}_test_outputs.npz"))['outputs']
        # Para multiclasse, outputs têm shape (n, 10) de probabilidades
        y_pred = np.argmax(outputs, axis=1)
        class_labels = CLASS_NAMES
    elif mode == 'Binary':
        (_, _), (x_test, y_test_binary) = prepare_binary_data(class_a, class_b)
        outputs = np.load(os.path.join(OUTPUT_DIR, f"{mode}_test_outputs.npz"))['outputs']
        y_pred = (outputs > 0.5).astype(int).squeeze()
        y_test = y_test_binary
        class_labels = [CLASS_NAMES[class_a], CLASS_NAMES[class_b]]

    # acurácia geral
    acc = accuracy_score(y_test, y_pred)

    # matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # rwelatório de classificação
    report = classification_report(y_test, y_pred, target_names=class_labels)

    # métricas em arquivos
    with open(os.path.join(OUTPUT_DIR, f"{mode}_accuracy.txt"), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    with open(os.path.join(OUTPUT_DIR, f"{mode}_classification_report.txt"), 'w') as f:
        f.write(report)

    plot_confusion_matrix(cm, class_labels, mode)

    print(f"Avaliação {mode} concluída. Acurácia: {acc:.4f}")


if __name__ == '__main__':

    # ex multiclasse
    history_mult = load_training_log('multiclass')
    plot_training_curves(history_mult, 'multiclass')
    evaluate_model('multiclass')

    # ex binário
    history_bin = load_training_log('binary')
    plot_training_curves(history_bin, 'binary')
    # class_a e class_b são carregados do config ou passados aqui
    from src.config import BINARY_CLASSES
    evaluate_model('binary', *BINARY_CLASSES)
