# https://keras.io/examples/generative/conditional_gan/
from tensorflow import keras
from keras import Model
from keras import metrics
import numpy as np
import tensorflow as tf
import os


class GAN_model(keras.Model):
    def __init__(self, discriminator, generator, batch_size, label_size, num_gen_per_dis=1, dicriminator_ouput=None):
        super().__init__()
        self.discriminator: type[Model] = discriminator
        self.generator: type[Model] = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.batch_size = batch_size
        self.label_size = label_size
        self.num_gen_per_dis = num_gen_per_dis
        self.dicriminator_ouput = dicriminator_ouput

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    # @ tf.function
    def train_step(self, data):

        real_images, labels, bacgrounds = data

        batch_size = self.batch_size if real_images.shape[0] is None else real_images.shape[0]

        random_labels = create_ranom_label(batch_size, self.label_size)
        random_labels = tf.expand_dims(random_labels, axis=-1)

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))

        generated_mask = self.generator([bacgrounds, random_labels, random_latent_vectors])

        generated_images = generated_mask
        #generated_images = tf.clip_by_value(generated_images, clip_value_min=0, clip_value_max=1)

        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_labels = tf.concat([random_labels, labels], axis=0)

        # Assemble labels discriminating real from fake images.
        if self.dicriminator_ouput is None:
            fake_real_labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )
        else:
            fake_real_labels = tf.concat(
                [tf.ones((batch_size, self.dicriminator_ouput[0], self.dicriminator_ouput[1], 1)), tf.zeros((batch_size, self.dicriminator_ouput[0], self.dicriminator_ouput[1], 1))], axis=0
            )

        # Train the discriminator with combined images (real and fake)
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_labels])
            d_loss = self.loss_fn(fake_real_labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        for _ in range(self.num_gen_per_dis):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))

            if self.dicriminator_ouput is None:
                misleading_labels = tf.zeros((batch_size, 1))
            else:
                misleading_labels = tf.zeros((batch_size, self.dicriminator_ouput[0], self.dicriminator_ouput[1], 1))

            random_labels = create_ranom_label(batch_size, self.label_size)
            random_labels = tf.expand_dims(random_labels, axis=-1)

            with tf.GradientTape() as tape:
                fake_mask = self.generator([bacgrounds, random_labels, random_latent_vectors])

                fake_images = fake_mask
                #fake_images = tf.clip_by_value(fake_images, clip_value_min=0, clip_value_max=1)
                predictions = self.discriminator([fake_images, random_labels])
                g_loss = self.loss_fn(misleading_labels, predictions)

            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    def generate_output(self, data):
        real_images, labels, bacgrounds = data

        batch_size = tf.shape(real_images)[0]
        random_labels = create_ranom_label(batch_size, self.label_size)
        random_labels = tf.expand_dims(random_labels, axis=-1)

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))

        generated_mask = self.generator([bacgrounds, random_labels, random_latent_vectors])

        generated_images = generated_mask
        return generated_images, generated_mask, random_labels, random_latent_vectors, bacgrounds

    # def save_weights(self, path):
    #     path_d = os.path.join(path, "discriminator.h5")
    #     path_g = os.path.join(path, "generator.h5")
    #     self.discriminator.save_weights(path_d)
    #     self.generator.save_weights(path_g)

    # def load_weights(self, path):
    #     path_d = os.path.join(path, "discriminator.h5")
    #     path_g = os.path.join(path, "generator.h5")
    #     self.discriminator.load_weights(path_d)
    #     self.generator.load_weights(path_g)


def create_ranom_label(batch_size, label_size):
    rows, cols = label_size
    result = np.zeros((batch_size, rows, cols), dtype=np.float32)
    random_indices = np.random.randint(0, rows, size=(batch_size, rows))
    result[np.arange(batch_size)[:, np.newaxis], random_indices, np.arange(cols)] = 1

    return result
