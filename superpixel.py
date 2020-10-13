"""Module to create super pixels, find centroids and new inputs to the next stages and estimate coherence
Created on 2020-04-28
File: superpixel.py
...
@author: Frincy Clement

"""


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

def superpixel(input):

    """
    Function to perform Stage I Super pixeling on input data

    Args:
    input: 4 channel input data of SAR images

    Returns:

    segments_fz,segments_slic : Super pixels using felzenszwalb and SLIC
    """

    img = input

    # Different values of model parameters for felzenszwalb Super pixeling
#     segments_fz = felzenszwalb(img, scale=20, sigma=0.2, min_size=20, multichannel=True) #for 1179
#     segments_fz = felzenszwalb(img, scale=15, sigma=0.2, min_size=10, multichannel=True) #for 2676
    segments_fz = felzenszwalb(img, scale=10, sigma=0.2, min_size=5, multichannel=True) #for 6367
#     segments_fz = felzenszwalb(img, scale=5, sigma=0.1, min_size=3, multichannel=True) #for 12853


     # Different values of model parameters for SLIC Super pixeling
#     segments_slic = slic(img, n_segments=100, compactness=1, sigma=1,multichannel=True)
#     segments_slic = slic(img, n_segments=2000, compactness=1, sigma=1,multichannel=True)
    segments_slic = slic(img, n_segments=5700, compactness=1, sigma=1,multichannel=True)

    # Diplaying the number of super pixels for both algorithms
    print(f"Felzenszwalb number of Super Pixels: {len(np.unique(segments_fz))}")
    print(f"SLIC number of Super Pixels: {len(np.unique(segments_slic))}")

    return segments_fz,segments_slic


def centroid(segments,input):
    """
    Function to find the centroid and individual channel values of super pixels

    Args:

    segments: Super pixels from State I Super pixeling
    input: 4 channel input data for InSAR dataset

    Returns:
    centroid: 4 channel Cluster centroids
    a1_list : Amplitude channel in super pixels for SAR Image 1
    a2_list : Amplitude channel in super pixels for SAR Image 2
    real_list: Real component in super pixels for Interferrogram
    imag_list: Imaginary component in super pixels for Interferrogram

    """
    centroids = []
    amp_slc1 = input[:,:,0]
    amp_slc2 = input[:,:,1]
    real_ifg_phase = input[:,:,2]
    imag_ifg_phase = input[:,:,3]
    coordinates = []

    # Finding the coordinates of pixels in each clusters
    for i in range(len(np.unique(segments))):
        indices = []
        for row_index,row in enumerate(segments):
            for col_index,item in enumerate(row):
                if(item==i):
                    indices.append([row_index,col_index])
        coordinates.append(indices)
    delta_list = []
    a1_list,a2_list,real_list,imag_list = [],[],[],[]
    sum=0

    # For each coordinates in clusters, finding the value from 4 channel input data
    for i in range(len(coordinates)):

        a1,a2,real,imag = [],[],[],[]
        for loc in coordinates[i]:
            x,y=loc[0],loc[1]
            a1.append(amp_slc1[x,y])
            a2.append(amp_slc2[x,y])
            real.append(real_ifg_phase[x,y])
            imag.append(imag_ifg_phase[x,y])
        sum += len(a1)
        a1_list.append(np.mean(a1))
        a2_list.append(np.mean(a2))
        real_list.append(np.mean(real))
        imag_list.append(np.mean(imag))

    # Stacking the 4 channels in the third dimension
    centroids = np.dstack((a1_list,a2_list,real_list,imag_list))

    return centroids,a1_list,a2_list,real_list,imag_list, coordinates



def highway_pixel(input, superPixel, highwayPixel):
    """
    Function to find the highway pixel label at pixel level

    Args:

    input: 4 channel input dataset
    superPixel: Super pixels from Stage I Super pixeling
    highwayPixel: Super pixels from Stage II Super pixeling

    Returns:

    new_input: highway labels decomposed to pixel level

    """

    new_input = np.zeros((input[:,:,0].shape),dtype=int)

    for row_index,row in enumerate(superPixel):
        for col_index,item in enumerate(row):
            superPixel_label = item
            highwayPixel_label = highwayPixel[superPixel_label]
            new_input[row_index][col_index] = highwayPixel_label

    return new_input

def centroid_single(segments,input):
    """
    Function to find the centroid of single dimensional set of clusters

    Args:

    segments: super pixels from the Stage I Super pixeling
    input: 4 channel input data

    Returns:

    centroids: 4 channel Cluster centroids
    centroid: 4 channel Cluster centroids
    a1_list : Amplitude channel in super pixels for SAR Image 1
    a2_list : Amplitude channel in super pixels for SAR Image 2
    real_list: Real component in super pixels for Interferrogram
    imag_list: Imaginary component in super pixels for Interferrogram
    """


    centroids = []
    amp_slc1 = input[:,:,0]
    amp_slc2 = input[:,:,1]
    real_ifg_phase = input[:,:,2]
    imag_ifg_phase = input[:,:,3]
    coordinates = []

    # Finding the coordinates of pixels in each clusters
    for i in range(len(np.unique(segments))):
        indices = []
        for loc,item in enumerate(segments):

            if(item==i):
                indices.append(loc)
        coordinates.append(indices)

    delta_list = []
    a1_list,a2_list,real_list,imag_list = [],[],[],[]
    sum=0

    # Finding the input channel values for each pixel location
    for i in range(len(coordinates)):

        a1,a2,real,imag = [],[],[],[]
        for loc in coordinates[i]:
            # print(loc)
            a1.append(amp_slc1[0,loc])
            a2.append(amp_slc2[0,loc])
            real.append(real_ifg_phase[0,loc])
            imag.append(imag_ifg_phase[0,loc])
        cx = np.mean(coordinates[i])
        centroids.append(cx)
        sum += len(a1)
        a1_list.append(np.mean(a1))
        a2_list.append(np.mean(a2))
        real_list.append(np.mean(real))
        imag_list.append(np.mean(imag))

    # Stacking the 4 channels in the third dimension
    centroids = np.dstack((a1_list,a2_list,real_list,imag_list))

    return centroids,a1_list,a2_list,real_list,imag_list, coordinates

def coherence_estimator(highwayToPixel, slc1,slc2):
    """
    Function to estimate the coherence value at high way pixel level

    Args:

    highwayToPixel: Highway pixel labels copied to pixel level
    slc1: SAR Image 1
    slc2: SAR Image 2

    Returns:

    delta_list: Coherence estimated using Maximum liklihood estimator

    """
    coordinates = []
    sum=0

    # Traversing from highway to pixel to find the coordinates
    for i in range(len(np.unique(highwayToPixel))):
        indices = []
        for row_index,row in enumerate(highwayToPixel):
            for col_index,item in enumerate(row):
                if(item==i):
                    indices.append([row_index,col_index])

        sum+=len(indices)
        coordinates.append(indices)
    delta_list = []

    # Finding the values in SAR images for pixels cluster-wise for applyin
    # Maximum liklihood estimator
    for i in range(len(coordinates)):

        z1,z2 = [],[]
        for coord in coordinates[i]:
            x,y=coord[0],coord[1]
            z1.append(slc1[x,y])
            z2.append(slc2[x,y])


        numerator,deno1,deno2 = 0,0,0
        # Applying maximum likihood estimator for each cluster
        delta = np.abs(np.sum(z1*np.conj(z2))/np.sqrt(np.sum(np.abs(z1)**2.)*np.sum(np.abs(z2)**2.)))
        delta_list.append(delta)

    return delta_list

def coherence_estimator_super(superToPixel,slc1,slc2):

    """
    Function to estimate the coherence value at super pixel level

    Args:

    superToPixel: Highway pixel labels copied to pixel level
    slc1: SAR Image 1
    slc2: SAR Image 2

    Returns:

    delta_list: Coherence estimated using Maximum liklihood estimator
    """
    coordinates = []
    sum=0

    # Traversing from super pixel to pixel level to find coordinates of clusters
    for i in range(len(np.unique(superToPixel))):
        indices = []
        for row_index,row in enumerate(superToPixel):
            for col_index,item in enumerate(row):
                if(item==i):
                    indices.append([row_index,col_index])
        sum+=len(indices)
        coordinates.append(indices)
    delta_list = []

    # For coordinates in each super pixels, finding coherence
    for i in range(len(coordinates)):

        z1,z2 = [],[]
        for coord in coordinates[i]:
            x,y=coord[0],coord[1]

            z1.append(slc1[x,y])
            z2.append(slc2[x,y])

        numerator,deno1,deno2 = 0,0,0

        # Calculating coherence for each clusters using maximum liklihood estimator
        delta = np.abs(np.sum(z1*np.conj(z2))/np.sqrt(np.sum(np.abs(z1)**2.)*np.sum(np.abs(z2)**2.)))
        delta_list.append(delta)

    print(np.asarray(delta_list).shape)
    return delta_list


def oneD_regionProp(segments):

    """
    Function to return the clusters grouped together for single array
    similar to regionprops algorithm from scikit-learn

    Args:

    segments: Highway pixels from the Stage II Super pixeling

    Returns:

    region: list of list with cluster names added in pixel locations
    """
    region = dict()
    unique = np.unique(segments)

    # Traversing through unique cluster labels
    for i in unique:
        region[i] = []
    c=0
    temp = []

    # Traversing through highway pixels
    for i in segments:
        print(i)
        region[i].append(c)
        c+=1
    return region
