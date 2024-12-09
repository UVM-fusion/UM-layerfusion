<div align="center">
<h2>Towards Maximizing the Sweet Spot for NLP Models</h2>
This repo provides the code implementation for the final project of the <b>[2024] Deep Learning & NLP course</b>.
</div>

## 📋Overview
Large NLP models face significant memory constraints, making it challenging to execute them effectively due to their increasing memory demands. To tackle this, we integrate **Unified Memory** with **Layer Fusion**—a method that allows programs to utilize more memory than typically available. This approach seeks to balance memory efficiency and performance, enabling large Transformer models to run seamlessly on a single processor with minimal performance degradation. The project evaluates the effectiveness of this strategy in supporting large-scale NLP models.


<div align="center">
    <img src="https://github.com/UVM-fusion/UVM-layerfusion/blob/main/assets/DLNLP_Overview.png" alt="Project_overview" height="400em"/>
</div>


## 🛠 Environment Setup
To conduct the experiments, it is necessary to install **PyTorch-UVM** configured to operate in a Unified Memory environment.<br/>
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

### ✅ Install Requirements 
``` bash
# install requirements
pip install -r requirements.txt
```


## ⚙️ How to Run Experiments
- We implemented five versions of GPT model layers in a Unified Memory system with Layer Fusion.
- To select the desired version of GPT model layers, modify the import statement for PrefetchGPT2LM at the top of the run_gpt2.py file.
``` bash
# run_gpt2.py

# 1. Baseline: Layers with only Unified Memory applied (Default)
from models.gpt2_prefetch.py import PrefetchGPT2LM

# 2. Fused (Dropout + LayerNorm) Layer
from dropoutlayernorm_gpt2_prefetch.py import PrefetchGPT2LM

# 3. Fused (Attention + LayerNorm) Layer
from attnlayernorm_gpt2_prefetch.py import PrefetchGPT2LM

# 4. Fused (Feed Forward + LayerNorm) Layer
from ffnlayernorm_gpt2_prefetch.py import PrefetchGPT2LM

# 5. Fused (Attention + Projection) Layer
from attnprojctn_gpt2_prefetch.py import PrefetchGPT2LM
```

- Each version of the layer can be executed on three sizes of GPT models:
``` bash
# 1. gpt2_1.5b (1.5B parameters)
$ python run_gpt2.py --model gpt2_1.5b --enable-prefetch --enable-cudnn-benchmark --num-streams 5 --warmups 5

# 2. gpt3_6.7b (6.7B parameters)
$ python run_gpt2.py --model gpt3_6.7b --enable-prefetch --enable-cudnn-benchmark --num-streams 5 --warmups 5

# 3. gpt3_13b (13B parameters)
$ python run_gpt2.py --model gpt3_13b --enable-prefetch --enable-cudnn-benchmark --num-streams 5 --warmups 5
```




## 📊 Experiment Results

실행환경


<div align="center">
    <img src="https://github.com/UVM-fusion/UVM-layerfusion/blob/main/assets/dlnlp_perf_eval.png" alt="dlnlp_perf_eval" height="200"/>
</div>

## 🌟 Project Members (Team09)

| <img width="200" src="https://user-images.githubusercontent.com/68412683/206727359-a653906e-0847-4702-a7e4-4c1ac532bd46.png"/> | <img width="200" src="https://github.com/UVM-fusion/UVM-layerfusion/blob/main/assets/user_image.png"/> |
| --- | --- |
| **Ji Yeong Yi** | **Jane Rhee** |
| --- | --- |
| AIX 4th | AIX 5th |
| --- | --- |
| jybyte@gmail.com | jrhee1122@ewhain.net |
