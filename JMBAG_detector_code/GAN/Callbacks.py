# https://stackoverflow.com/questions/47676248/accessing-validation-data-within-a-custom-callback

import keras
from keras import callbacks
import tensorflow as tf

from pathlib import Path
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from AnnotationData import AnnotationData
import numpy as np
import glob
import shutil


class ShowGeneratedImages(callbacks.Callback):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def on_train_begin(self, logs=None):
        generated_images, generated_mask, random_labels, random_latent_vectors, backrounds = self.model.generate_output(self.data[0])
        generated_images = tf.clip_by_value(generated_images, 0, 1)
        showImages(backrounds, generated_mask, generated_images, random_labels)

    def on_epoch_end(self, epoch, logs=None):

        generated_images, generated_mask, random_labels, random_latent_vectors, backrounds = self.model.generate_output(self.data[0])
        generated_images = tf.clip_by_value(generated_images, 0, 1)
        showImages(backrounds, generated_mask, generated_images, random_labels)


def showImages(backgrounds, masks, combined, labels, rows=4):
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3*rows))

    for i in range(rows):
        labelStr = AnnotationData.convertAnnotationToStr(labels[i])

        # Plot the background image
        axes[i, 0].imshow(tf.squeeze(backgrounds[i]), cmap='gray')
        # Plot the mask image
        axes[i, 1].imshow(tf.squeeze(masks[i]), cmap='gray')
        axes[i, 1].set_title(labelStr)

        # Plot the combined image
        axes[i, 2].imshow(tf.squeeze(combined[i]), cmap='gray')

        # Remove ticks and labels for all subplots
        for j in range(3):
            axes[i, j].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot (optional)
    plt.show()
