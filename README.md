# CaCo

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/CaCo-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-MIT-green">
</a>   

This is a modified version of original CaCo code with following changes:

1- The original code was meant to run on multiple GPUs. Although you can actually use that code for a single-GPU setup, it seemed to me that it would add some delay and perhaps computational overhead. The code in this repo can be run on a single GPU easily.

2- The architecture of conventional ResNets, as proposed in the paper "Deep Residual Learning for Image Recognition," was modified to meet the needs of the CIFAR-10 dataset (other options may be employed given sufficient processing power).

3- The transformations were adjusted to match the CIFAR-10 images.

4- I tried to find the best possible hyperparameters.

5- Added the option to simulate the behavior of multiple GPUs using split batch normalization.

6- Added "batch_shuffle_single_gpu" to shuffle queries and make use of shuffleBN on single gpu ( must be used with splitbatchnorm)

CaCo is a contrastive-learning based self-supervised learning methods, which is published in IEEE-T-PAMI.

Copyright (C) 2022 Xiao Wang, Yuhang Huang, Dan Zeng, Guo-Jun Qi

License: MIT for academic use.

To Contact authors of the paper: Xiao Wang (wang3702@purdue.edu), Guo-Jun Qi (guojunq@gmail.com)

To contact me: erfankolsoumian@gmail.com



## Installation  
CUDA version should be 10.1 or higher. 
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:maple-research-lab/CaCo.git && cd CaCo
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch>=1.7.1
pip install torchvision>=0.8.2
pip install numpy>=1.19.5
pip install Pillow>=5.1.0
pip install tensorboard>=1.14.0
pip install tensorboardX>=1.7
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n CaCo python=3.7.1
conda activate CaCo
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate CaCo
conda deactivate(If you want to exit) 
```

## Usage

### Single-Crop Unsupervised Pre-Training

```
python3 main.py --type=0 --lr=0.75 --lr_final=0.003 --memory_lr=3.0 --memory_lr_final=3.0 --cluster=8192 --mem_t=0.1 --data= [path] --dist_url=tcp://localhost:10001 --batch_size=512 --wd=1.1e-4 --mem_wd=0 --moco_dim=128 --moco_m=0.99 --moco_t=0.1 --mlp_dim=512 --dataset stl10 --epochs=800 --warmup_epochs=0 --nodes_num=1 --workers=4 --cos=2 --gpu=0 --arch resnet18 --world_size 1 --moco_m_decay=1 --multiprocessing_distributed=0 --rank=0 --mem_momentum=0.9 --ad_init=1 --knn_batch_size=512 --multi_crop=0 --knn_freq=5 --knn_neighbor=5

```


### Linear Classification
With a pre-trained model, we can easily evaluate its performance on desired dataset with:
```
python linear.py  -a resnet18 --lr 0.25 --batch-size 512 \
  --pretrained [your checkpoint path] \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0 --data [dataset path]
```


### Performance:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>network</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">Crop</th>
<th valign="bottom">Symmetrical<br/>Loss</th>
<th valign="bottom">CaCo<br/>KNN acc.</th>

<!-- TABLE BODY -->
<tr><td align="left">ResNet-18</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">No</td>
<td align="center">87.34</td>

</tr>
<tr><td align="left">ResNet-18</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">Yes</td>
<td align="center">88.99</td>

</tr>
</tbody></table>


## Citation:
[CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning](https://arxiv.org/abs/2203.14370).  
[CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning](https://doi.org/10.1109/TPAMI.2023.3262608). 
```
@article{wang2022caco,
  title={CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning },
  author={Wang, Xiao and Huang, Yuhang and Zeng, Dan and Qi, Guo-Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
```

