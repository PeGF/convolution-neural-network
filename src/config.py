# Dados
DATASET_NAME = "FASHION_MNIST"

CNN_CONFIG = {
    "input_shape": (28, 28, 1),
    "conv_layers": [
        {"filters": 32, "kernel_size": (3, 3), "activation": "relu"},
        {"filters": 64, "kernel_size": (3, 3), "activation": "relu"}
    ],
    "pool_size": (2, 2),
    "dense_units": 128,
    "dropout_rate": 0.5,
    "activation": "relu"
}

# Hiperparâmetros da Rede
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Número de classes
NUM_CLASSES_MULTICLASS = 10  # Multiclasse 0 a 9
NUM_CLASSES_BINARY = 2  # Binário: 2 classes (sempre 2, obviamente)

# 0 - T-shirt/top
# 1 - Trouser
# 2 - Pullover
# 3 - Dress
# 4 - Coat
# 5 - Sandal
# 6 - Shirt
# 7 - Sneaker
# 8 - Bag
# 9 - Ankle boot
BINARY_CLASSES = (0, 6)

# Diretórios de saída
OUTPUT_DIR = "outputs"
WEIGHTS_INITIAL_FILE = f"{OUTPUT_DIR}/initial_weights.h5"
WEIGHTS_FINAL_FILE = f"{OUTPUT_DIR}/final_weights.h5"
ERROR_LOG_FILE = f"{OUTPUT_DIR}/error_log.csv"
OUTPUTS_FILE = f"{OUTPUT_DIR}/outputs.csv"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Descritores
USE_DESCRIPTORS = False  # False -> imagens brutas | True -> descritores
DESCRIPTOR_TYPE = "hog"  # Opções: 'hog', 'lbp', 'haar'

RANDOM_SEED = 42  # Reprodutibilidade