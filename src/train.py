import os
import numpy as np
import tensorflow as tf
import pandas as pd

from src.data_loader import (
    prepare_multiclass_data,
    prepare_binary_data,
    split_train_validation
)
from src.model_builder import build_cnn_model
from src.config import *


def save_weights(model, filename):
    """
    Salva os pesos do modelo no formato numpy (.npz).
    """
    weights = model.get_weights()
    np.savez_compressed(filename, *weights)


def save_training_logs(history, filename):
    """
    Salva o histórico de treinamento (loss, accuracy, etc) em CSV.
    """
    pd.DataFrame(history.history).to_csv(filename, index=False)


def save_model_outputs(model, x_test, filename):
    """
    Salva as saídas do modelo para os dados de teste.
    """
    outputs = model.predict(x_test)
    np.savez_compressed(filename, outputs=outputs)


def train_task(mode):
    """
    Executa o treinamento da CNN.
    
    Parâ^metro:
    - mode: 'Multiclass' ou 'Binary'
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    '''
    Dados
    '''
    if mode == "Multiclass":
        (x_train, y_train), (x_test, y_test) = prepare_multiclass_data()
        num_classes = NUM_CLASSES_MULTICLASS
    elif mode == "Binary":
        (x_train, y_train), (x_test, y_test) = prepare_binary_data()
        num_classes = NUM_CLASSES_BINARY
    else:
        raise ValueError("Modo inválido. Use 'multiclass' ou 'binary'.")

    x_train, x_val, y_train, y_val = split_train_validation(x_train, y_train)

    '''
    Modelo
    '''
    model = build_cnn_model(num_classes)

    '''
    Salva pesos iniciais
    '''
    initial_weights_file = os.path.join(OUTPUT_DIR, f"{mode}_initial_weights.npz")
    save_weights(model, initial_weights_file)

    '''
    Treinamento
    '''
    history = model.fit(
        x_train, y_train,
        epochs=NUM_EPOCAS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=2
    )

    '''
    Salva pesos finais
    '''
    final_weights_file = os.path.join(OUTPUT_DIR, f"{mode}_final_weights.npz")
    save_weights(model, final_weights_file)

    '''
    Salva o modelo completo
    '''
    model_save_path = os.path.join(OUTPUT_DIR, f"{mode}_model.keras")
    model.save(model_save_path)

    '''
    Salva logs de treinamento
    '''
    logs_file = os.path.join(OUTPUT_DIR, f"{mode}_training_log.csv")
    save_training_logs(history, logs_file)

    '''
    Salva saídas do modelo
    '''
    outputs_file = os.path.join(OUTPUT_DIR, f"{mode}_test_outputs.npz")
    save_model_outputs(model, x_test, outputs_file)

    print(f"Treinamento {mode} concluído. Arquivos salvos na pasta {OUTPUT_DIR}.")
