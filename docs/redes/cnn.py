#CNN – Classificação de Imagens MNIST
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten

# Ajustar dados
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Modelo CNN
cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat))
