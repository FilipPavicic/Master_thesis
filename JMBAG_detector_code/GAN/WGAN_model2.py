# https://keras.io/examples/generative/conditional_gan/
from tensorflow import keras
from keras import Model
from keras import metrics
import numpy as np
import tensorflow as tf
import os


class GAN_model(keras.Model):
    def __init__(self, discriminator, generator, batch_size, label_size, num_dis_updates=1, num_gen_updates=1, dicriminator_ouput=None, alfa=10, lambda_gp=10):
        super().__init__()
        self.discriminator: type[Model] = discriminator
        self.generator: type[Model] = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.identity_loss_tracker = keras.metrics.Mean(name="identity_loss")
        self.batch_size = batch_size
        self.label_size = label_size
        self.num_dis_updates = num_dis_updates
        self.num_gen_updates = num_gen_updates
        self.dicriminator_ouput = dicriminator_ouput
        self.alfa = alfa
        self.lambda_gp = lambda_gp

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker, self.identity_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, identity_loss):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.identity_loss = identity_loss

    def gradient_penalty(self, batch_size, real_images, fake_images, real_labels, fake_labels):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        diff_label = fake_labels - real_labels
        interpolated_labels = diff_label + alpha * diff_label

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, interpolated_labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @ tf.function
    def train_step(self, data):

        real_images, labels, backgrounds = data

        batch_size = self.batch_size if real_images.shape[0] is None else real_images.shape[0]

        # Update discriminator multiple times
        for _ in range(self.num_dis_updates):

            random_labels = create_ranom_label(batch_size, self.label_size)
            random_labels = tf.expand_dims(random_labels, axis=-1)

            random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))

            generated_mask = self.generator([backgrounds, random_labels, random_latent_vectors], training=True)

            generated_images = backgrounds - generated_mask
        #generated_images = tf.clip_by_value(generated_images, clip_value_min=0, clip_value_max=1)

            combined_images = tf.concat([generated_images, real_images], axis=0)
            combined_labels = tf.concat([random_labels, labels], axis=0)
            # Train the discriminator with combined images (real and fake)
            with tf.GradientTape() as tape:
                predictions = self.discriminator([combined_images, combined_labels])
                d_loss_real = -tf.reduce_mean(predictions[:batch_size])
                d_loss_fake = tf.reduce_mean(predictions[batch_size:])
                d_loss = d_loss_real + d_loss_fake

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, generated_images, labels, random_labels)

                d_loss += self.lambda_gp * gp

            gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
            # Gradient clipping
            # clipped_gradients = [tf.clip_by_value(g, -0.01, 0.01) for g in gradients]
            self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        for _ in range(self.num_gen_updates):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.label_size[0], self.label_size[1], 1))

            random_labels = create_ranom_label(batch_size, self.label_size)
            random_labels = tf.expand_dims(random_labels, axis=-1)

            with tf.GradientTape() as tape:
                fake_mask = self.generator([backgrounds, random_labels, random_latent_vectors])
                fake_images = backgrounds - fake_mask
                predictions = self.discriminator([fake_images, random_labels])
                wg_loss = -tf.reduce_mean(predictions)

                # mask_real_label = self.generator([backgrounds, labels, random_latent_vectors])
                # image_real_label = backgrounds - mask_real_label
                # image_real_label = tf.clip_by_value(image_real_label, clip_value_min=0, clip_value_max=1)
                # identity_loss = self.identity_loss(real_images, image_real_label) * self.alfa

                # g_loss = wg_loss + identity_loss
                g_loss = wg_loss

            gradients = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(wg_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.identity_loss_tracker.update_state(0.0)
        return {

            "d_loss": self.disc_loss_tracker.result(),
            "g_wg_loss": self.gen_loss_tracker.result(),
            "g_id_loss": self.identity_loss_tracker.result(),

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

        generated_images = bacgrounds - generated_mask
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


def gradients(inputs, outputs):
    gradients = tf.gradients(outputs, inputs)
    # Normalize gradients (important for stability)
    gradients_normalized = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return gradients_normalized


def random_weighted_average(real_samples, generated_samples):
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform(shape=(batch_size, 1, 1, 1))
    interpolated_samples = alpha * real_samples + (1 - alpha) * generated_samples
    return interpolated_samples


def gradient_penalty(gradients):
    gradient_penalty = tf.reduce_mean(tf.square(gradients - 1.0))
    return gradient_penalty
