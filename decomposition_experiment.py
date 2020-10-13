""" Module with different decomposition functions defined for each of the six experiments
Created on 2020-04-28
File: decomposition_experiment.py
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
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
from numpy.linalg import norm

def decomposition_highway(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 1: Inverse distance decomposition

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

    # Normalizing centroids and inputs to be between 0 and 1
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
            if distance == 0:
                coh = coherence[c]
            else:
                coh = coherence[c]*(1-distance)
            decom_highway_coh[point] = coh
        c+=1


    return decom_highway_coh

def decomposition_super(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 1: inverse distance decomposition

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

            if distance == 0:
                coh = coherence[c]
            else:
                coh = coherence[c]*(1-distance)
            decom_super_coh[x][y] = coh
        c+=1

    return decom_super_coh

def decomposition_highway1(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 2: Differential Decomposition with level-specific weight

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
            coh = (coherence[c]*(1-distance))**(1/2)
            decom_highway_coh[point] = coh
        c+=1


    return decom_highway_coh


def decomposition_super1(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 2: Differential Decomposition with level-specific weight

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
    for i in range (0, 300):
        new = []
        for j in range (0, 300):
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

            coh = (coherence[c]*(1-distance))
            decom_super_coh[x][y] = coh
        c+=1

    return decom_super_coh


def decomposition_highway2(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 3: Differential Decomposition with distance-specific weight

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

            if distance < 0.3:
                coh = (coherence[c]*(1-distance))**(1/2)
            else:
                coh = (coherence[c]*(1-distance))**(2)
            decom_highway_coh[point] = coh
        c+=1


    return decom_highway_coh

def decomposition_super2(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 3: Differential Decomposition with distance-specific weight

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
    for i in range (0, 300):
        new = []
        for j in range (0, 300):
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

            if distance < 0.3:
                coh = (coherence[c]*(1-distance))
            else:
                coh = (coherence[c]*(1-distance))**(2)
            decom_super_coh[x][y] = coh
        c+=1

    return decom_super_coh

def decomposition_highway3(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 4,5,6: Passing highway pixels without weights

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


def decomposition_super3(centroid, highway, coherence,coordinates,input):
    """
    Function to perform Experiment 4,5,6: Passing super pixels without weights

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
    for i in range (0, 300):
        new = []
        for j in range (0, 300):
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
