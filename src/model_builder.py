# src/model_builder.py

import tensorflow as tf
from keras import layers, models, optimizers
from src.config import CNN_CONFIG, LEARNING_RATE


def build_cnn_model(num_classes):
    """
    Cria uma CNN com base nas configurações definidas em config.py.

    Parâmetros:
    - num_classes: número de classes da saída (10 para multiclasse, 2 para binário)

    Retorna:
    - modelo compilado pronto para treino
    """
    input_shape = CNN_CONFIG["input_shape"]

    # Instanciação do modelo sequencial
    model = models.Sequential()

    '''
    Camadas Convolucionais
    '''

    for idx, conv_layer in enumerate(CNN_CONFIG["conv_layers"]):
        if idx == 0:
            # Primeira camada convolucional precisa definir input_shape
            model.add(layers.Conv2D(
                filters=conv_layer["filters"],
                kernel_size=conv_layer["kernel_size"],
                activation=conv_layer["activation"],
                input_shape=input_shape
            ))
        else:
            # Próximas camadas não precisam definir input_shape
            model.add(layers.Conv2D(
                filters=conv_layer["filters"],
                kernel_size=conv_layer["kernel_size"],
                activation=conv_layer["activation"]
            ))

        # Adiciona pooling após cada convolução
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    '''
    Flatten para conectar na camada densa
    '''
    model.add(layers.Flatten())

    '''
    Camada densa escondida
    '''
    model.add(layers.Dense(CNN_CONFIG["dense_units"], activation='relu'))

    '''
    Dropout para reduzir overfitting
    '''
    model.add(layers.Dropout(CNN_CONFIG["dropout_rate"]))

    '''
    Camada de saída
    '''
    if num_classes == 2:
        # Classificação binária
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        # Classificação multiclasse
        model.add(layers.Dense(num_classes, activation='softmax'))

    '''
    Compilação do modelo
    '''
    if num_classes == 2:
        loss_function = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss_function = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)

    return model
