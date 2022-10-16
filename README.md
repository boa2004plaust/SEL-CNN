# A Simple Efficient Light-weighted CNN for 5G LOS/NLOS Identification
Code for A Simple Efficient Light-weighted CNN for 5G LOS/NLOS Identification

Author: Yasong Zhu, Jiabao Wang, Bing Xu*, Peng Liu, Wangdong Qi.

Last Update: 16/10/2022

CITATION:

If you use this code in your research, please cite:

	@ARTICLE {SEL-CNN,
	author    = "Yasong Zhu, Jiabao Wang, Bing Xu, Peng Liu, Wangdong Qi",
	title     = "A Simple Efficient Light-weighted CNN for 5G LOS/NLOS Identification",
	journal   = {IEEE XXXX},
	year      = {2022},
	}
  
  
# DATA Preparation
You can download the data from [Baidu](https://pan.baidu.com/s/1BFoogq4PqT2mU8H9j2w4Qg) (Extracting code: 6txw).

# Training and Testing
## 1. Configuration

Python-3.7

Pytorch-1.7.1

numpy-1.24.1

scipy-1.7.3


## 2. Training
Run the bash file "train_models.sh"

## 3. Testing
Run the bash file "test.sh"

# Reference

@article{DBLP:journals/icl/CuiGHTC21,
  author    = {Zhichao Cui and
               Yufang Gao and
               Jing Hu and
               Shiwei Tian and
               Jian Cheng},
  title     = {{LOS/NLOS} Identification for Indoor {UWB} Positioning Based on Morlet
               Wavelet Transform and Convolutional Neural Networks},
  journal   = {{IEEE} Commun. Lett.},
  volume    = {25},
  number    = {3},
  pages     = {879--882},
  year      = {2021},
  url       = {https://doi.org/10.1109/LCOMM.2020.3039251},
  doi       = {10.1109/LCOMM.2020.3039251},
}
