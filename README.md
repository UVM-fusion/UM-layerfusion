# Towards Maximizing the Sweet Spot for NLP Models
This repo provides the code implementation for the final project of the [2024] Deep Learning & NLP course.

## 📋Overview
Large NLP models face significant memory constraints, making it challenging to execute them effectively due to their increasing memory demands. To tackle this, we integrate **unified memory** with **layer fusion**—a method that allows programs to utilize more memory than typically available. This approach seeks to balance memory efficiency and performance, enabling large Transformer models to run seamlessly on a single processor with minimal performance degradation. The project evaluates the effectiveness of this strategy in supporting large-scale NLP models.


<div align="center">
    <img src="https://github.com/UVM-fusion/UVM-layerfusion/blob/main/assets/DLNLP_Overview.png" alt="Project_overview" height="400em"/>
</div>

## 🛠 Environment Setup
To conduct the experiments, it is necessary to install PyTorch-UVM configured to operate in a Unified Memory environment.<br/>
(This section is inspired by the setup guide from **https://github.com/kooyunmo/cuda-uvm-gpt2**)

### ✅ PyTorch-UVM Prerequisites
- Ubuntu 18.04
- anaconda3
- cuda-11.0
- cudnn 8.0.4 for cuda-11.0
- correct environment variables

### ✅ PyTorch-UVM Installation
``` bash
git clone --recursive https://github.com/kooyunmo/cuda-uvm-gpt2
cd cuda-uvm-gpt2/pytorch-uvm
git checkout uvm

# create new conda environment
conda create -n uvm-pytorch python=3.8 -y
conda activate uvm-pytorch

# environment variables, we need this setting for every installation and experiment
export CUDA_HOME=<YOUR_CUDA_11.0_PATH>
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDNN_LIB_DIR=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include/
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# install dependencies
# ensure prerequisites for pytorch build
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing -y
conda install -c pytorch magma-cuda110 -y

# install onnx
conda install -c conda-forge onnx -y

# downgrade protobuf ([why?](https://github.com/onnx/onnx/issues/2434))
conda install -c conda-forge protobuf=3.9 -y

# ensure prerequisites for caffe2 build
pip install future

# run setup.py
BUILD_TEST=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_NUMA=0 USE_MPI=0 python setup.py install
``` 

### ✅ install requirements 
``` bash
# install requirements
pip install -r requirements.txt
```





## 👟 Run Experiments



## 📊 Experiment Results



## 🌟 Project members (Team09)

| <img width="200" src="https://user-images.githubusercontent.com/68412683/206727359-a653906e-0847-4702-a7e4-4c1ac532bd46.png"/> | <img width="200" src="https://user-images.githubusercontent.com/68412683/206727359-a653906e-0847-4702-a7e4-4c1ac532bd46.png"/> |
| --- | --- |
| **Ji Yeong Yi** | **Jane Rhee** |
| AIX 4th | AIX 5th |
| jybyte@gmail.com | jrhee1122@ewhain.net |
