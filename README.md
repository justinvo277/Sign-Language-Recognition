# Two Stream S3D Architecture for Word Level Sign Language Recognition


## Introduce

This repository accompanies the paper [Two Stream S3D Architecture for Word Level Sign Language Recognition](https://dl.acm.org/doi/10.1145/3654522.3654559). The article addresses sign language recognition at the word level based on the Separable 3D CNN (S3D) model. We propose a low-cost model because we recognize the potential for future use of identification systems on handheld devices. We have conducted experiments on many different data sets and achieved the expected results.

## Two Stream S3D Architecture

![Architecture](images/architecture.png)

## Dataset

We tested on three different datasets including: [Large-Scale Multimodal Turkish Signs (AUTSL)](https://ieeexplore.ieee.org/abstract/document/9210578), [Large-Scale Dataset for Word-Level American Sign Language (WLASL)](https://github.com/dxli94/WLASL), and [A Dataset for Argentinian Sign Language (LSA64)](https://facundoq.github.io/datasets/lsa64/). You can explore and download the data on our link provided in this repository.

## Data Folder

To make it easier for you to use your custom data, we describe in detail the structure of the folder containing the data as follows:

```
WLASL_100/
│
├── test/
│   └── # Testing data goes here
│
├── train/
│   ├── africa/
│   ├── clothes/
│   ├── dark/
│   ├── give/
│   └── later/
│   └── # Training data is organized by class in separate folders
│
└── val/
    └── # Validation data goes here
```



