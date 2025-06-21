'''
Hiperparâmetros da CNN
'''

BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

# Diretório de saída
OUTPUT_DIR = "outputs"

# Arquitetura da CNN
CNN_CONFIG = {
    "input_shape": (28, 28, 1),  # Formato da imagem do MNIST
    "num_classes_multiclass": 10,
    "num_classes_binary": 2,
    
    # Definição das camadas convolucionais
    "conv_layers": [
        {"filters": 32, "kernel_size": (3, 3), "activation": "relu"},
        {"filters": 64, "kernel_size": (3, 3), "activation": "relu"}
    ],

    "dense_units": 128, # Definição da camada densa final

    "dropout_rate": 0.5, # Dropout para regularização
}

'''
Parametros
'''

RANDOM_SEED = 42 

# Classes do MNIST
BINARY_CLASSES = (3, 8) # Trocar dps para testar

'''
Pastas pra salvar arquivos
'''

PATHS = {
    "models": "./models/",
    "outputs": "./outputs/",
    "logs": "./outputs/logs/",
    "plots": "./outputs/plots/",
    "predictions": "./outputs/predictions/"
}
