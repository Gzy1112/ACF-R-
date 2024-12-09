# ACF-R+: An Asymmetry-sensitive Method for Image-Text Retrieval enhanced by Cross-Modal Fusion and Re-ranking based on Contrastive Learning

## Introduction

This is the source code of "ACF-R+: An Asymmetry-sensitive Method for Image-Text Retrieval enhanced by Cross-Modal Fusion and Re-ranking based on Contrastive Learning". 

This manuscript has been submitted to Neurocomputing.

## Requirements

We recommended the following dependencies:

- Python 3.9
- PyTorch 2.0.1
- NumPy 1.24.3
- torchtext 0.15.2
- pycocotools 2.0.6
- nltk 3.8.1
- opencv-python 4.5.5.64
- h5py 3.8.0
- torchvision 0.15.1+cu118
- Pillow 9.3.0
- torchtext 0.15.1
- tensorboard 2.12.2
- tensorboard-logger 0.1.0
  ......

## Description

- opts.py: parameter settings
- data.py: data preprocessing
- model.py: the model of ACF-R+
- train.py: model training
- evaluation.py: model evaluation
- vocab.py: vocabulary wrapper
- rerank.py: reranking

## Data

Public data can be downloaded from "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives", F. Faghri, D. J. Fleet, J. R. Kiros, S. Fidler, Proceedings of the British Machine Vision Conference (BMVC), 2018. (BMVC Spotlight)

## Train Models

Modify the data_name, data_path and feature path in the `opt.py` file. Then run `train.py`:

```
python train.py
```

## Evaluate Models

Modify the model_path and data_path in the `evaluation.py` file. Then run `evaluation.py`:

```
python evaluation.py
```
