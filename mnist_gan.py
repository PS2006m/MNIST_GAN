import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 5000        
SAVE_EVERY = 1000        
RESULTS_DIR = 'gan_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
IMG_ROWS, IMG_COLS, CHANNELS = 28, 28, 1
BUFFER_SIZE = 60000

(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5  # scale to [-1,1]
x_train = np.expand_dims(x_train, axis=-1)            # shape (N,28,28,1)
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_generator(latent_dim):
    model = models.Sequential(name='generator')
    model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((7, 7, 128)))  # 7x7x128

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())                # 14x14x64

    model.add(layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'))
    # 28x28x1 output in range [-1,1]
    return model

def build_discriminator(img_shape):
    model = models.Sequential(name='discriminator')
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # logits
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss

def generator_loss(fake_logits):
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)

generator = build_generator(LATENT_DIM)
discriminator = build_discriminator((IMG_ROWS, IMG_COLS, CHANNELS))

gen_optimizer = optimizers.Adam(1e-4)
disc_optimizer = optimizers.Adam(1e-4)

checkpoint_dir = './gan_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
os.makedirs(checkpoint_dir, exist_ok=True)

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_logits)
        disc_loss = discriminator_loss(real_logits, fake_logits)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def save_images(generator, step, n=16):
    noise = tf.random.normal([n, LATENT_DIM])
    gen_imgs = generator(noise, training=False)
    gen_imgs = (gen_imgs + 1.0) * 127.5  # back to [0,255]
    gen_imgs = gen_imgs.numpy().astype(np.uint8)

    cols = 4
    rows = n // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            axs[r, c].imshow(gen_imgs[idx, :, :, 0], cmap='gray')
            axs[r, c].axis('off')
            idx += 1
    plt.suptitle(f'Step {step}')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'generated_{step:05d}.png')
    plt.savefig(path)
    plt.close(fig)

def train(dataset, steps):
    step = 0
    for epoch in range(999999):  # effectively run until step >= steps
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            step += 1

            if step % 100 == 0:
                print(f"Step {step}: gen_loss={g_loss.numpy():.4f}, disc_loss={d_loss.numpy():.4f}")

            if step % SAVE_EVERY == 0:
                save_images(generator, step)
                checkpoint.save(file_prefix=checkpoint_prefix)
                print(f"Saved sample & checkpoint at step {step}")

            if step >= steps:
                return

train(dataset, steps=EPOCHS)
save_images(generator, step=EPOCHS)
print("Training finished. Samples in", RESULTS_DIR) 
