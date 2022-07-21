#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning Studio                                                                #
# Version    : 0.1.0                                                                               #
# Filename   : /regression.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/DLStudio                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday July 21st 2022 12:33:56 pm                                                 #
# Modified   : Thursday July 21st 2022 12:52:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------ #


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend()
