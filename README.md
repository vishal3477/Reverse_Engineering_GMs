# Reverse_Engineering_GMs
Official Pytorch implementation of paper "Reverse Engineering of Generative Models: Inferring Model Hyperparameters from Generated Images"
![alt text](https://github.com/vishal3477/Reverse_Engineering_GMs/blob/main/image/teaser.png?raw=true)
## Prerequisites

- PyTorch 1.5.0
- Numpy 1.14.2
- Scikit-learn 0.22.2

## Getting Started

## Datasets
For reverse enginnering: 
- Download the dataset for 100 Gdenerative models from https://drive.google.com/drive/folders/1uRlrJDgV8yzTVqx62p8IAIHd7oDfhFYv?usp=sharing
- For leave out experiment, put the training data in train folder and leave out models data in test folder
- For testing on custom images, put the data in test folder.

For deepfake detection:
- Download the CelebA/LSUN dataset

For image_attribution:
- Generate the data for four different GAN models as specified in https://github.com/ningyu1991/GANFingerprints/

## Training
- Provide the train and test path in respective codes as sepecified below
- Provide the model path to resume training
- Run the code

For reverse engineering:
- Run reverse_eng.py

For deepfake detection: 
- Run deepfake_detection.py

For image attribution:
- Run image_attribution.py

## Testing using pre-trained models
- Provide the test path in respective codes as sepecified below
- Download the pre-trained models from https://drive.google.com/drive/folders/1bzh9Pvr7L-NyQ2Mk-TBSlSq4TkMn2anB?usp=sharing
- Provide the model path in the code
- Run the code

For reverse engineering:
- Run reverse_eng_test.py

For deepfake detection: 
- Run deepfake_detection_test.py

For image attribution:
- Run image_attribution_test.py
