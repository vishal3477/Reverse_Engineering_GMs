# <p align=center>`Model Parsing Dataset`</p>




## 1. Overview
We make the first attempt to study the model parsing problem. we assemble a list of 116 publicly available GMs, each having 1000 images genearted using the publicly avalable code. The entire list of GMs can be found in the supplementary. 

We further document the model hyperparameters for each GM as reported in their papers. Specifically, we investigate two aspects: network architecture and training loss functions. We form a super-set of 15 network architecture parameters (e.g., number of layers, normalization type) and 10 different loss function types

<p align="center">
    <img src="../image/Screenshot 2023-07-31 at 7.58.48 PM.png"/> <br />
</p>

## 2. Dataset images

The images can be found on the google drive using the the [link](https://drive.google.com/file/d/1bAmC_9aMkWJB_scGvOOWvNeLa9FBoMUr/view). 

Some sample images for each GM are shown below:

<p align="center">
    <img src="../image/Screenshot 2023-07-31 at 8.01.15 PM.png"/> <br />
</p>

## 3. Ground-truth files
We provide the numpy files for cluster ground truth and deviation ground-truth, both for network architecture and loss function for each generative model. 

## 4. Diffusion Model Dataset
The dataset for 7 diffusion models used for our experiemnts can be downloaded from [link](https://drive.google.com/drive/folders/1S9Y2LtP7ZHNDOjuR7mp5tx4LxuhG5EPy?usp=drive_link).

The ground truth files are available in the directory. 

## 5. License
The dataset can be used for research purposes only. Please contact me via [email](asnanivi@msu.edu) for commercial usage permission.
