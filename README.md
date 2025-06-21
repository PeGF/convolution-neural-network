# Classificação de Dígitos com Redes Neurais Convolucionais (CNN)

Este projeto foi desenvolvido como parte de um trabalho acadêmico da disciplina de Inteligência Artificial do curso de Sistemas de Informação (USP), com o objetivo de implementar e treinar uma Rede Neural Convolucional (CNN) utilizando o framework TensorFlow/Keras.

## Objetivo

O trabalho consiste em:

- Implementar uma Rede Neural Artificial do tipo Convolutional Neural Network (CNN) com múltiplas camadas ocultas e uma camada densa ao final.
- Realizar dois tipos de tarefas de classificação utilizando o conjunto de dados MNIST:
  - **Classificação Multiclasse:** A rede deve ser capaz de identificar corretamente os 10 dígitos (0 a 9) do MNIST.
  - **Classificação Binária:** A rede deve ser treinada para distinguir entre duas classes específicas (por exemplo, dígito "0" contra dígito "1") ou combinações de classes.

## Abordagem

Além do treinamento direto com os dados brutos (imagens), o trabalho permite a utilização de técnicas de extração de características, como:

- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Transformada Wavelet de Haar

## Saídas Geradas

Durante o desenvolvimento, o projeto gera os seguintes arquivos de saída:

- Arquivo com os hiperparâmetros utilizados na rede neural.
- Arquivo contendo os pesos iniciais da rede.
- Arquivo contendo os pesos finais após o treinamento.
- Arquivo com o erro da rede em cada época (loss curve).
- Arquivo com as predições realizadas pela rede no conjunto de teste.

## Bibliotecas Utilizadas

- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-image (para extração de descritores, quando aplicável)
- OpenCV

## Fonte dos Dados

O conjunto MNIST está disponível em:

- [TensorFlow Datasets - MNIST](https://www.tensorflow.org/datasets/catalog/mnist?hl=pt-br)
