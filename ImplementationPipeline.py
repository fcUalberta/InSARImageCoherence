"""Program to Implement the Pipeline of Proposed Method

Created on 2020-04-28
File: ImplementationPipeline.py
...
@author: Frincy Clement
"""

# ## Importing Libraries and dependencies

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
from numpy.linalg import norm
import os

from superpixel import superpixel, centroid, highway_pixel,coherence_estimator, oneD_regionProp, centroid_single
from highwayDecomp import decomposition_highway, decomposition_super
from utils import readFloatComplex, readShortComplex, readFloat

# Initializing variables
WIDTH = 300
PATCH_SIZE = [300,300]  # patch size
STRIDE = 150     # Stride
NUM_FILES = 4

# Number of clusters in Stage II Super Pixeling
k=75

# Defining path for output visuals
path = r"outputs\pipeline"

# Defining path for input dataset
base = r"data2"
filelist = os.listdir(base)


# ## Function to calculate Stage II Super pixels

# In[2]:


def calculate_highway(superpixel,input,k,slc1,slc2):
    """
    Pipeline to calculate highway pixels from input super pixels

    Args:

    superpixel: Stage I super pixels
    input: 4 channels input data
    k: Number of clusters for Stage II super pixeling
    slc1: First SAR Image
    slc2: Second SAR Image

    Returns:

    coh: Coherence decomposed to pixel level
    new_input: Labels decomposed to pixel level
    """
    # Finding the Cluster centroids along each channel and stacked
    centroids_super,a1_list,a2_list,real_list,imag_list,coordinates_super = centroid(superpixel,input)

    # Stacking the input for Stage II Super pixeling
    input_super = np.dstack((a1_list,a2_list,real_list,imag_list))

    # Stage II super pixeling using K-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(input_super.reshape(-1,4))
    highway = kmeans.labels_

    # Passing the highway cluster labels to pixel-level
    new_input = highway_pixel(input,superpixel,highway)

    # Finding the centroid of highway pixel and each channel separately
    centroids_highway,a1_list,a2_list,real_list,imag_list, coordinates_high = centroid_single(highway ,input_super)

    # Coherence Estimation using Maximum Liklihood Estimat
    highway_coh = coherence_estimator(new_input,slc1,slc2)

    # Decomposing highway pixel to superpixel
    decom_super_coh = decomposition_highway(centroids_highway, highway,highway_coh,coordinates_high,input_super)

    # Decomposing super pixel to pixel-level
    coh = decomposition_super(centroids_super,superpixel,decom_super_coh,coordinates_super,input)

    return coh, new_input


# ## Function to calculate evaluation metrics

# In[3]:


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


# ## Function to display visual comparison results

# In[4]:


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


# ## Function to implement Boxcar (sliding window)

# In[5]:


def sliding_window(input, slc1,slc2,stepSize, windowSize):
    """
    Function to perform boxcar function on the input

    Args:

    input: 4 channel input data on which boxcar needs to be performed
    slc1: First SAR Image
    slc2: Second SAR Image
    stepSize: Stride requuired for the sliding window
    windorSize: Size of sliding window

    Returns:
    coh : Coherence estimate using sliding window
    """

    coh = np.zeros((300,300))

    # slide a window across the image
    for y in range(0, 300, stepSize):
        for x in range(0, 300, stepSize):

            # yield the current window
            window = input[y:y + windowSize[1], x:x + windowSize[0],:]

            # Getting the window on SAR Image 1 and 2
            z1 = slc1[y:y + windowSize[1], x:x + windowSize[0]]
            z2 = slc2[y:y + windowSize[1], x:x + windowSize[0]]

            # Coherence estimate using maximum liklihood estimator
            delta = np.abs(np.sum(z1*np.conj(z2))/np.sqrt(np.sum(np.abs(z1)**2.)*np.sum(np.abs(z2)**2.)))

            # Assigning pixel-wise coherence
            coh[x,y] = delta

    return np.transpose(coh)


# ## Function to implement the pipeline of the proposed application (For one sample input)

# In[6]:


def pipeline(filelist, method):
    """
    Function to read the files, implement the pipeline

    Inputs:

    filelist: list of input files
    method: Method for Stage I Super pixeling
    """
    mse_list, rmse_list, ssim_list = [],[],[]
    mse_list_sliding, rmse_list_sliding, ssim_list_sliding = [],[],[]
    coh_list, coh3vg_list, coh_list_sliding = [],[],[]
    for i in range(0,int(len(filelist)/NUM_FILES)):

        # Loading first dataset

#         IFG_PATH = os.path.join(base,str(i)+"slc1_"+str(i)+"slc2.noisy")
#         COH_PATH = os.path.join(base,str(i)+"slc1_"+str(i)+"slc2.filt.coh")
#         SLC1_PATH = os.path.join(base,str(i)+"slc1.rslc")
#         SLC2_PATH = os.path.join(base,str(i)+"slc2.rslc")

#         ifg = readFloatComplex(IFG_PATH, WIDTH)
#         coh_3vg = readFloat(COH_PATH, WIDTH)
#         slc1 = readFloatComplex(SLC1_PATH, WIDTH)
#         slc2 = readFloatComplex(SLC2_PATH, WIDTH)

        # Loading second dataset
        ifg = np.load(os.path.join(base,str(i)+".ifg.npy"))
        coh_3vg = np.load(os.path.join(base,str(i)+".coh.npy"))
        slc1 = np.load(os.path.join(base,str(i)+".slc1.npy"))
        slc2 = np.load(os.path.join(base,str(i)+".slc2.npy"))


        # Creating the 4-channel input

        # Dim-0
        amp_slc1 = np.abs(slc1)

        # Dim-1
        amp_slc2 = np.abs(slc2)

        # Phase of Ifg
        phase_ifg = np.angle(ifg)

        # Force amp to one
        phase_bar_ifg = 1*np.exp(1j*phase_ifg)

        # Dim-2
        real_ifg_phase = np.real(phase_bar_ifg)
        # Dim-3
        imag_ifg_phase = np.imag(phase_bar_ifg)

        # Stacking each dimension in each channel
        input = np.dstack((amp_slc1,amp_slc2,real_ifg_phase,imag_ifg_phase))

        width = len(input)
        L,H = PATCH_SIZE[0],PATCH_SIZE[1]

        count=0
        switch2= 0

        # Sliding through the window
        for y in range(0, width, STRIDE):

            if (y+H > width):
                y = width - H
                switch2=1

            switch1=0
            for x in range(0, width, STRIDE):

                # Adjusting x and y values for the last patch horizontally and vertically
                if (x+L > width):
                    x = width - L
                    switch1 = 1
                print("X values", x,x+L)
                print("Y values",y, y+H)

                # yield the current window
                window = input[y:y + H, x:x + L,:]

                # Getting the window on SAR Image 1 and 2
                z1 = slc1[y:y + H, x:x + L]
                z2 = slc2[y:y + H, x:x + L]
                coh_window = coh_3vg[y:y + H, x:x + L]

                # Stage I of Super pixeling
                superpixel_fz,superpixel_slic=superpixel(window)

                if method == 'slic':
                    super = superpixel_slic
                else:
                    super = superpixel_fz

                # Calculating highway pixels and coherence
                coh, highway_labels = calculate_highway(super,window, k,z1,z2)

                # Calculating evaluation metrics
                print("Proposed Methods Results\n")
                mse,rmse,ssim = calculate_metrics(coh_window, coh)

                # Appending coherence values of all inputs to a list for saving
                coh3vg_list.append(coh_window)
                coh_list.append(coh)

                # Appending metrics values of all inputs to a list
                mse_list.append(mse)
                rmse_list.append(rmse)
                ssim_list.append(ssim)

                # Displaying the results
                plt = display_result(coh_window, coh)
                plt.savefig(os.path.join(path,str(i)+r"_patch"+str(count)+"_Result"))
                plt.show()

                # Sliding window 7x7

                # Padding the input with zeros for sliding window
                data = np.pad(window, ((3,3), (3,3), (0, 0)), 'constant')

                # Padding the SAR images with zeros for sliding window
                s1 = np.pad(z1, ((3,3), (3,3)), 'constant')
                s2 = np.pad(z2, ((3,3), (3,3)), 'constant')

                # Calculating coherence using sliding window of size 7x7
                coh_sliding = sliding_window(data,s1,s2,1,[7,7])

                # Appending coherence values sliding window to a list for saving
                coh_list_sliding.append(coh_sliding)

                # Appending metrics values of sliding window to a list
                print("Boxcar Results\n")
                mse_sliding,rmse_sliding,ssim_sliding = calculate_metrics(coh_window, coh_sliding)

                # Appending metrics values of sliding window to a list
                mse_list_sliding.append(mse_sliding)
                rmse_list_sliding.append(rmse_sliding)
                ssim_list_sliding.append(ssim_sliding)

                # Displaying the results
                plt = display_result(coh_window, coh_sliding)
                plt.savefig(os.path.join(path,"Sliding"+str(i)+r"_patch"+str(count)+"_Result"))
                plt.show()

                count += 1


                if (width==H):
                    break
                if(switch1 ==1):
                    break
            if(switch2==1):
                break
        print("Input number",i)



    return coh_list,coh3vg_list, coh_list_sliding, mse_list,rmse_list,ssim_list,mse_list_sliding,rmse_list_sliding,ssim_list_sliding,i


# ## Main Function

# In[7]:


if __name__ == '__main__':
    """
    Implementing the pipeline
    """

    coh_list,coh3vg_list, coh_list_sliding, mse_list,rmse_list,ssim_list,mse_list_sliding,rmse_list_sliding,ssim_list_sliding,i = pipeline(filelist, 'slic')


# In[8]:


print("Average MSE",np.mean(mse_list))
print("Average RMSE",np.mean(rmse_list))
print("Average SSIM",np.mean(ssim_list))


print("Sliding Window")

print("Average MSE",np.mean(mse_list_sliding))
print("Average RMSE",np.mean(rmse_list_sliding))
print("Average SSIM",np.mean(ssim_list_sliding))


# In[11]:


# Saving all the coherence files

np.save('coh_slic_'+str(i), coh_list)
np.save('coh3vg_slic_'+str(i),coh3vg_list)
np.save('coh_sliding_slic_'+str(i),coh_list_sliding)


# # End of Program
