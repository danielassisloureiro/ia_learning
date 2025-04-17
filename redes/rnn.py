#RNN (LSTM) – Classificação de Sentimentos IMDB
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, LSTM

# Dados sequenciais
max_features = 5000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Modelo RNN com LSTM
rnn = Sequential([
    Embedding(max_features, 32, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
