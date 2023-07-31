# <p align=center>`Model Parsing (IEEE TPAMI)`</p>
Official Pytorch implementation of our **T-PAMI** paper "Reverse Engineering of Generative Models: Inferring Model Hyperparameters from Generated Images".

The paper and supplementary can be found at [Arxiv](https://arxiv.org/abs/2106.07873) 

> **Authors:** 
> [Vishal Asnani](https://vishal3477.github.io/), 
> [Xi Yin](https://xiyinmsu.github.io/), 
> [Tal Hassner](https://mmcheng.net/) &
> [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html).

## 1. :fire: NEWS :fire:

- [2023/07/29] Our Paper is accepted to **Transactions on Pattern Analysis and Machine Intelligence**!!
- [2023/07/29] The version 3 of the code is released!
- [2022/06/30] The version 2 of the code is released!
- [2022/01/06] Our codebase for model parsing is released is released!

## 2. Overview
<p align="center">
    <img src="./image/teaser_resized.png"/> <br />
</p>

## 3. Training/testing


### Prerequisites

- PyTorch 1.5.0
- Numpy 1.14.2
- Scikit-learn 0.22.2


### Datasets

We collect a large scale dataset comprising of fake images images genearted by 116 generative models. Please visit [link]() for more details. 
For reverse enginnering: 
- Download the dataset and the ground truth files from the [dataset website](). 
- For leave out experiment, put the training data in train folder and leave out models data in test folder
- For testing on custom images, put the data in test folder.

For deepfake detection:
- Download the CelebA/LSUN dataset

For image_attribution:
- Generate 110,000 images for four different GAN models as specified in https://github.com/ningyu1991/GANFingerprints/
- For real images, use 110,000 of CelebA dataset.
- For training: we used 100,000 images and remaining 10,000 for testing.

### Training
- Provide the train and test path in respective codes as sepecified below. 
- Provide the model path to resume training
- Run the code

For reverse engineering, run:
```
python reverse_eng.py
```

For deepfake detection, run: 
```
python deepfake_detection.py
```

For image attribution, run:
```
python image_attribution.py
```

### Testing using pre-trained models
- Provide the test path in respective codes as sepecified below
- Download the pre-trained models from https://drive.google.com/drive/folders/1bzh9Pvr7L-NyQ2Mk-TBSlSq4TkMn2anB?usp=sharing
- Provide the model path in the code
- Run the code

For reverse engineering, run:
```
python reverse_eng_test.py
```
For deepfake detection, run: 
```
python deepfake_detection_test.py
```
For image attribution, run: 
```
python image_attribution_test.py
```

If you would like to use our work, please cite:
```
@misc{asnani2023reverse,
      title={Reverse Engineering of Generative Models: Inferring Model Hyperparameters from Generated Images}, 
      author={Vishal Asnani and Xi Yin and Tal Hassner and Xiaoming Liu},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2023}
}
```
