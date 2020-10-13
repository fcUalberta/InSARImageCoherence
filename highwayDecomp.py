"""
Module to perform decomposition from highway pixel to super pixels
in the implementation pipeline based on the learnings from experiments

Created on 2020-04-28
File: highwayDecomp.py
...
@author: Frincy Clement

"""

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops
from operator import itemgetter
from numpy.linalg import norm
import os


def decomposition_highway(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 4 for pipeline: Passing
    highway pixels without weights

    Args:

    centroid: Cluster centroid of highway pixels
    highway: Highway pixels after Stage II Super pixeling
    coherence: Coherence value at highway pixel level
    coordinates: Coordinates of pixels in each highway clusters
    input: 4 channel input data

    Returns:
    decom_highway_coh: Coherence estimate passed from highway to super pixel
    """
    c = 0
    decom_highway_coh = [0]*len(highway)

    # Normalizing centroids and input_sl
    input_min = input.min(axis=(0, 1), keepdims=True)
    input_max = input.max(axis=(0, 1), keepdims=True)
    input_norm = (input - input_min)/(input_max - input_min)

    c_min = centroid.min(axis=(0, 1), keepdims=True)
    c_max = centroid.max(axis=(0, 1), keepdims=True)
    c_norm = (centroid - c_min)/(c_max - c_min)

    # Traversing through each cluster coordinates to calculate
    # distance between pixels and cluster coordinates
    # To assign coherence value to super pixel level

    for cluster in coordinates:
        clusterCenter = c_norm[0][c]
        for point in cluster:
            superPixel = input_norm[0][point]
            distance = norm(clusterCenter-superPixel)
            coh = coherence[c]
            decom_highway_coh[point] = coh
        c+=1


    return decom_highway_coh

def decomposition_super(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 4: Passing super pixels without weights

    Args:

    centroid: Cluster centroid of super pixels
    highway: Super pixels after Stage I Super pixeling
    coherence: Coherence value at super pixel level
    coordinates: Coordinates of pixels in each highway clusters
    input: 4 channel input data

    Returns:
    decom_super_coh: Coherence estimate passed from super pixel to pixel level
    """

    c = 0

    decom_super_coh = []
    for i in range (0, len(input)):
        new = []
        for j in range (0, len(input)):
            new.append(0)
        decom_super_coh.append(new)

    # Normalizing centroids and input_sl
    input_min = input.min(axis=(0, 1), keepdims=True)
    input_max = input.max(axis=(0, 1), keepdims=True)
    input_norm = (input - input_min)/(input_max - input_min)

    c_min = centroid.min(axis=(0, 1), keepdims=True)
    c_max = centroid.max(axis=(0, 1), keepdims=True)
    c_norm = (centroid - c_min)/(c_max - c_min)

    # Traversing through each cluster coordinates to calculate
    # distance between pixels and cluster coordinates
    # To assign coherence value to pixel level

    for cluster in coordinates:
        clusterCenter = c_norm[0][c]
        for point in cluster:
            x,y = point[0],point[1]
            superPixel = input_norm[x,y]
            distance = norm(clusterCenter-superPixel)
            coh = coherence[c]

            decom_super_coh[x][y] = coh
        c+=1

    return decom_super_coh
