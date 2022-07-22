#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning Studio                                                                #
# Version    : 0.1.0                                                                               #
# Filename   : /image.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/DLStudio                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday July 21st 2022 01:20:37 pm                                                 #
# Modified   : Thursday July 21st 2022 01:56:51 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import tensoflow as tf
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------ #
# Create a function for plotting a random image along with its prediction


def plot_random_image(model, images, true_labels, classes):
    """Picks a random image, plots it and labels it with a predicted and truth label.

    Args:
        model: a trained model (trained on data similar to what's in images).
        images: a set of random images (in tensor form).
        true_labels: array of ground truth labels for images.
        classes: array of class names for images.

    Returns:
        A plot of a random image from `images` with a predicted class label from `model`
        as well as the truth class label from `true_labels`.
    """
    # Setup random integer
    i = random.randint(0, len(images))

    # Create predictions and targets
    target_image = images[i]
    pred_probs = model.predict(
        target_image.reshape(1, 28, 28)
    )  # have to reshape to get into right size for model
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # Plot the target image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Change the color of the titles depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Add xlabel information (prediction/true label)
    plt.xlabel(
        "Pred: {} {:2.0f}% (True: {})".format(
            pred_label, 100 * tf.reduce_max(pred_probs), true_label
        ),
        color=color,
    )  # set the color to green or red


# ------------------------------------------------------------------------------------------------ #
def plot_multiple_images(images, labels):
    """Plots multiple images and their class labels. Adapted from Tensorflow Keras Classification
    Tutorial at https://www.tensorflow.org/tutorials/keras/classification

    Args:
        images (list): List of normalized images.
        labels (list): List of training labels corresponding to the images
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()
