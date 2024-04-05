#! /opt/homebrew/bin/fish

mkdir -p datasets

# Fashion MNIST
mkdir -p datasets/fashionmnist
kaggle datasets download zalando-research/fashionmnist -p datasets/fashionmnist --unzip
