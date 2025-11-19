#!/bin/bash
# DermFormer Environment Setup Script

echo "Creating DermFormer conda environment..."

# Create conda environment with Python 3.7
conda create -n dermformer python=3.7 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dermformer

# Install PyTorch with CUDA 11.5 support
echo "Installing PyTorch..."
conda install pytorch==1.12.1 torchvision==0.13.0 torchaudio==0.12.1 cudatoolkit=11.5 -c pytorch -c conda-forge -y

# Install conda packages
echo "Installing conda packages..."
conda install -c conda-forge \
    numpy=1.21.6 \
    pandas=1.3.5 \
    pillow=9.2.0 \
    scikit-learn=1.0.2 \
    scikit-image=0.19.3 \
    matplotlib=3.5.3 \
    pyyaml=6.0 \
    tqdm=4.66.4 \
    imageio=2.34.1 \
    -y

# Install pip packages
echo "Installing pip packages..."
pip install timm==0.6.13
pip install einops==0.6.1
pip install albumentations==1.3.1
pip install transformers==4.29.2
pip install tokenizers==0.13.1
pip install datasets==2.13.1
pip install optuna==3.6.1
pip install yacs==0.1.8
pip install tensorboard==1.14.0
pip install pytorch-model-summary==0.1.1
pip install torchinfo==1.8.0
pip install pydot==1.4.2
pip install opencv-python==4.6.0.66
pip install seaborn==0.12.2

# Install problematic packages
echo "Installing additional optimization packages..."
pip install torch-optimizer
pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git
pip install torchviz

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate dermformer"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
