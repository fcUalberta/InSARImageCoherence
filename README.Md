# InSAR Image Coherence using Multi-Stage Super pixeling

## Overview
Interferometric Synthetic Aperture Radar (InSAR) is a satellite-based imaging technique which has been used to learn about earth’s surface and sub-surface movements. It can measure earth’s displacements by comparing phase information from the Synthetic Aperture Radar (SAR) images taken at different points in time. But, due to high level of noises from scatterers, the wrapped phases are distorted. Estimating the coherence of SAR image pair becomes rather important in this situation to denoise them and extract useful information from it.

In our research, traditional methods of oversegmentation called Super Pixeling was performed in multiple levels, to capture the local as well as non-local coherence of images. The initial level of super pixeling was performed using SLIC and Felzenszwalb’s algorithm and the second level using Kmeans clustering algorithm. We performed our tests on two datasets with 0.1 to 0 left to right and 0.3 fixed noise levels respectively. Our results show that, on the dataset with higher noise, our proposed method with best tuned parameters gave a mean-squared error(MSE) of 0.0157 which is 1.9 times better than the widely used traditional method called Boxcar (sliding window) which has an MSE of 0.0302. Also, it presented 30% more structural similarity of the coherence estimates compared to that of Boxcar.

Our proposed method has outperformed even the Convolutional Neural Network(CNN) learning-based approach which gave a higher mean-squared error of 0.0299. Thus, our proposed approach estimates coherence between two SAR images, in a more robust and accurate way, compared to the existing traditional as well as learning-based state-of-the-art methods.

## [Read the full Research Report here](https://github.com/fcUalberta/InSARImageCoherence/blob/main/misc/InSAR%20Image%20Coherence%20Project%20Report.pdf)

## Implementation structure

The implementation was done in set of stages:
•	Pre-processing dataset: Converting the SAR images and Inteferogram into 4 channel input which consists of: (i) Amplitude of first SAR image, (ii) Amplitude of second SAR image, (iii) Real component of phase from the InSAR, and (iv)Imaginary component of phase from the InSAR.
•	Stage 1 of Super pixeling: Performed using Super pixeling algorithm of Felzenszwalb and SLIC to create super pixels from pixels
•	Stage 2 of Super pixeling: Performed using K-Means Clustering to create highway pixels from super pixels
o	A set of experiments were performed altering the parameters for the super pixeling and clustering algorithms in Stage I and II to find out the best combination of parameters.
•	Coherence Estimation at highway pixel level
•	Decomposition of highway pixel coherence back to pixel level.
o	A set of experiments were performed to find the weighting function to decompose the highway pixel coherence to pixel level
•	Comparison with Ground truth using Mean squared error and SSIM


## Project Structure
![GitHub Logo](/misc/ProjectStructure.PNG)

## Requirements
Software Requirements: Refer requirements.txt
Hardware Requirements: GPU not required, higher processor for faster computation


## How to run the application
1. Download the code folder as  zip and extract the files
2. Make sure you have the software requirements mentioned in requirements.txt
3. Change the path of execution to the project folder
4. The files can either be opened in jupyter notebook (ipynb) or run as python files
5. To run the ipynb files, you need to have Jupyter Notebook installed or can use Google Colab
6. To run python files, type python filename.py in the command line
	Eg: python Implementation_pipeline.py

