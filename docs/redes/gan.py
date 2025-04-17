#GAN – Geração de Dígitos Sintéticos
import tensorflow as tf
import numpy as np

# Dados MNIST normalizados
(train_images, _), _ = mnist.load_data()
train_images = train_images.reshape(-1, 784).astype("float32") / 255.0
BUFFER_SIZE = 60000
BATCH_SIZE = 128

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator
def make_generator():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(100,)),
        Dense(784, activation="sigmoid")
    ])
    return model

# Discriminator
def make_discriminator():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(784,)),
        Dense(1, activation="sigmoid")
    ])
    return model

generator = make_generator()
discriminator = make_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Treinamento simplificado de uma etapa
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise)

        real_output = discriminator(images)
        fake_output = discriminator(generated_images)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                    cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Treinamento (resumido)
for epoch in range(5):
    for image_batch in dataset:
        train_step(image_batch)
