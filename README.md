# 1. Install necessary packages
!pip install tensorflow tensorflow_datasets matplotlib

# 2. Import libraries
import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow.keras import layers

# 3. Upload the ZIP file containing your image pairs
uploaded = files.upload()
zip_file = list(uploaded.keys())[0]

# 4. Extract images from the ZIP
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

# 5. Load and preprocess images
def load_image_pairs(folder):
    input_images = []
    target_images = []
    filenames = sorted(os.listdir(folder))

    inputs = [f for f in filenames if f.startswith('input')]
    targets = [f for f in filenames if f.startswith('target')]

    for inp, tar in zip(sorted(inputs), sorted(targets)):
        input_path = os.path.join(folder, inp)
        target_path = os.path.join(folder, tar)

        input_img = tf.image.decode_jpeg(tf.io.read_file(input_path))
        target_img = tf.image.decode_jpeg(tf.io.read_file(target_path))

        input_img = tf.image.resize(input_img, [256, 256]) / 255.0
        target_img = tf.image.resize(target_img, [256, 256]) / 255.0

        input_images.append(input_img)
        target_images.append(target_img)

    return tf.data.Dataset.from_tensor_slices((input_images, target_images)).batch(1)

dataset = load_image_pairs('/content/dataset')

# 6. Define the Generator (U-Net)
def Generator():
    inputs = layers.Input(shape=[256, 256, 3])
    down1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    down1 = layers.LeakyReLU()(down1)

    down2 = layers.Conv2D(128, 4, strides=2, padding='same')(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU()(down2)

    up1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(down2)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.ReLU()(up1)
    up1 = layers.Concatenate()([up1, down1])

    up2 = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(up1)

    return tf.keras.Model(inputs=inputs, outputs=up2)

# 7. Define the Discriminator (PatchGAN)
def Discriminator():
    input_img = layers.Input(shape=[256, 256, 3], name='input_image')
    target_img = layers.Input(shape=[256, 256, 3], name='target_image')

    x = layers.concatenate([input_img, target_img])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)

    return tf.keras.Model(inputs=[input_img, target_img], outputs=x)

generator = Generator()
discriminator = Discriminator()

# 8. Define loss functions
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (100 * l1_loss)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

# 9. Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# 10. Training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

# 11. Train the model
import time
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    start = time.time()
    for input_image, target in dataset:
        train_step(input_image, target)
    print(f"Time taken for epoch: {time.time() - start:.2f} sec")

# 12. Display the output
def generate_images(model, test_input, target):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15, 5))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Target Image', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

# Display for one image
for inp, tar in dataset.take(1):
    generate_images(generator, inp, tar)# Task-3
