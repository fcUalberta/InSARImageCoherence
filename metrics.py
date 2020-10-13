"""Module to prepare evaluation metrics results and its visual comparisons

Created on 2020-04-28
File: metrics.py
...
@author: Frincy Clement
"""

from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(coh_3vg,coh):
    """
    Function to calculate MSE and SSIM

    Inputs:
    coh_3vg: Ground truth of coherence
    coh: Estimated coherence

    Returns:
    mses: Mean Squared error
    rmse: Root mean-squared error
    ssim: Structural Similarity Index
    """
    mses=mean_squared_error(coh_3vg,coh)
    print("MSE", mses)
    print("RMSE",np.sqrt(mses))
    (score, diff) = compare_ssim(np.array(coh_3vg), np.array(coh), full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    return mses,np.sqrt(mses),score


def display_result(coh_3vg,coh):
    """
    Function to display visual comparison

    Inputs:
    coh_3vg: Ground truth of coherence
    coh: Estimated coherence

    Returns:
    matplotlib object for displaying

    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(coh_3vg)
    ax[0].set_title("Ground Truth")
    ax[1].imshow(coh, vmin=np.array(coh).min(), vmax = np.array(coh).max())
    ax[1].set_title("Estimated Level Coh")
    return plt
