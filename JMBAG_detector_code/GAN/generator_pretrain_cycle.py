# https://keras.io/examples/generative/conditional_gan/
from tensorflow import keras
from keras import Model
from keras import metrics
import numpy as np
import tensorflow as tf
import os


class GAN_model(keras.Model):
    def __init__(self, generator, batch_size, label_size,):
        super().__init__()
        self.generator: type[Model] = generator
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.batch_size = batch_size
        self.label_size = label_size

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compile(self,  optimizer, loss):
        super().compile()
        self.optimizer = optimizer
        self.loss = loss

    # @ tf.function
    def train_step(self, data):

        real_images, labels, bacgrounds = data

        batch_size = self.batch_size if real_images.shape[0] is None else real_images.shape[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))
        random_labels = create_ranom_label(batch_size, self.label_size)
        random_labels = tf.expand_dims(random_labels, axis=-1)

        with tf.GradientTape() as tape:
            up_mask = self.generator([real_images, random_labels, random_latent_vectors])
            up_image = real_images + up_mask

            down_mask = self.generator([up_image, labels, random_latent_vectors])
            down_image = up_image + down_mask
            down_image = tf.clip_by_value(down_image, clip_value_min=0, clip_value_max=1)
            loss = self.loss(real_images, down_image)

        grads = tape.gradient(loss, self.generator.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result(),
        }

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    def generate_output(self, data):
        real_images, labels, bacgrounds = data

        batch_size = tf.shape(real_images)[0]
        random_labels = create_ranom_label(batch_size, self.label_size)
        random_labels = tf.expand_dims(random_labels, axis=-1)

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))

        generated_mask = self.generator([real_images, random_labels, random_latent_vectors])

        generated_images = real_images + generated_mask
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
