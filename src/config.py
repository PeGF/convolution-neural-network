# Dados
DATASET_NAME = "MNIST"
INPUT_SHAPE = (28, 28, 1)  # Formato das imagens do MNIST


# Hiperparâmetros da Rede
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Número de classes
NUM_CLASSES = 10  # Multiclasse 0 a 9
# Binário está definido na main como 1

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

RANDOM_STATE = 42 