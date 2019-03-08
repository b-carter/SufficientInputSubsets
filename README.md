# Understanding Decisions with Sufficient Input Subsets

This repository contains code for the following paper:

***"What made you do this? Understanding black-box decisions with sufficient input subsets." Brandon Carter, Jonas Mueller, Siddhartha Jain, David Gifford. 2018.***  [[arxiv]](https://arxiv.org/abs/1810.03805)

In this work, we propose *sufficient input subsets*, minimal subsets of input features whose values alone suffice for the model to reach the same decision.
We also extract general principles that globally govern the model's decision making by clustering such input patterns that appear across many data points.
Our approach is entirely model-agnostic.


***Note: If you are looking to use the SIS method, we recommend taking a look at Google's implementation in the [google-research repository](https://github.com/google-research/google-research/tree/master/sufficient_input_subsets) (which only requires Python/NumPy, and contains extensive documentation and an interactive tutorial notebook). The code in this repository was used in the experiments presented in the paper, primarily for development. The SIS method was developed in the Gifford Lab at MIT CSAIL.***


##### Datasets

We explore our method to text, image, and genomic data:
* **Multi-aspect sentiment sentiment** of beer reviews from BeerAdvocate. 
* **Predicting transcription factor (TF) binding** in genomic data.
* **Classification of handwritten digits** using MNIST. 

See the `data/` directory for more details and links to obtain the datasets.


##### Directories

* `notebooks/`: Jupyter notebooks for model training, applying SIS and alternative methods, and analysis on beer review data; aggregate analysis of all TF binding datasets; MNIST
* `data/`: raw data for the various datasets
* `trained_models/`: trained (Keras) LSTM models for beer reviews
* `packages/`: respositories for integrated gradients (applied to Keras-based models) and Levenshtein distance
