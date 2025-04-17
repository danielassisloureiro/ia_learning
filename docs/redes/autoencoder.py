#Autoencoder – Compressão de Dígitos MNIST
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Dados normalizados
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

# Arquitetura Autoencoder
input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train_flat, x_train_flat, epochs=5, validation_data=(x_test_flat, x_test_flat))