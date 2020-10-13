InSAR Image Coherence using Multi-Stage Super pixeling

About
=====
This project is a part of Ualberta Masters program Course project for Computer Vision

Overview
========
Using conventional methods of oversegmentation called Super Pixeling, applied in multiple stages to get hierarchical segments and decomposed back to pixel level, we proposed a pipeline to estimate InSAR Image Coherence, which performs 1.9 times better than the widely used Boxcar method as well as CNN learning-based method.

Implementation structure
========================

The implementation was done in set of stages:
•	Pre-processing dataset: Converting the SAR images and Inteferogram into 4 channel input which consists of: (i) Amplitude of first SAR image, (ii) Amplitude of second SAR image, (iii) Real component of phase from the InSAR, and (iv)Imaginary component of phase from the InSAR.
•	Stage 1 of Super pixeling: Performed using Super pixeling algorithm of Felzenszwalb and SLIC to create super pixels from pixels
•	Stage 2 of Super pixeling: Performed using K-Means Clustering to create highway pixels from super pixels
o	A set of experiments were performed altering the parameters for the super pixeling and clustering algorithms in Stage I and II to find out the best combination of parameters.
•	Coherence Estimation at highway pixel level
•	Decomposition of highway pixel coherence back to pixel level.
o	A set of experiments were performed to find the weighting function to decompose the highway pixel coherence to pixel level
•	Comparison with Ground truth using Mean squared error and SSIM


Project Structure
=================

|------------------------------------------------
|------------------------------------------------
|--- SOURCE CODE
|---Experiments_Felzenszwalb.ipynb (or .py) # Experiments performed with Felzenszwalb's algorithm in Stage 1 Super pixeling
|---Experiments_SLIC.ipynb (or .py)         # Experiments performed with SLIC algorithm in Stage 1 Super pixeling
|--------decomposition_experiment.py        # Support Module with different decomposition function for each experiments
|---Implementation_pipeline.ipynb (or .py)  # Implementation of full pipeline for the best tuned parameters
|--------highwayDecomp.py                   # Support Module with best tuned decomposition function for implementation pipeline
|---superpixel.py                           # Support Module to create super pixels, find centroids and new inputs to the next stages and estimate coherence
|---metrics.py                              # Support Module to prepare evaluation metrics results and its visual comparisons
|---utils.py                                # Support Module to read the SAR images and interferrogram in the SLC format
|
|---HTML FILES (Source Code)
|---Experiments_Felzenszwalb.html           # Experiments performed with Felzenszwalb's algorithm in Stage 1 Super pixeling (HTML FILE)
|---Experiments_SLIC.html                   # Experiments performed with SLIC algorithm in Stage 1 Super pixeling (HTML FILE)
|---Implementation_pipeline.html            # Implementation of full pipeline for the best tuned parameters (HTML FILE)
|
|---FOLDERS
|---data1                                   # First dataset with 0.1 to 0 left to right noise (One sample set)
|---data2                                   # Second dataset with 0.3 fixed level noise. (One sample set)
|---outputs\fz                              # Output visuals with Felzenszwalb's algorithm in the first stage of pixeling (For One sample set)
|---outputs\slic                            # Output visuals with SLIC algorithm in the first stage of pixeling (For One sample set)
|---outputs\pipeline                        # Output visuals with SLIC algorithm and best tuned parameters added to the pipeline (For One sample set)
||-----------------------------------------------
|-----------------------------------------------


Requirements
============

Software Requirements: Refer requirements.txt
Hardware Requirements: GPU not required, higher processor for faster computation


How to run the application
==========================
P.S: OPEN THE HTML FILES IN YOUR BROWSER FOR EASY VIEWING OF THE IPYNB FILES WITH RESULTS AND VISUALS.

1. Download the code folder as  zip and extract the files
2. Make sure you have the software requirements mentioned in requirements.txt
3. Change the path of execution to the project folder
4. The files can either be opened in jupyter notebook (ipynb) or run as python files
5. To run the ipynb files, you need to have Jupyter Notebook installed or can use Google Colab
6. To run python files, type python filename.py in the command line
	Eg: python Implementation_pipeline.py

