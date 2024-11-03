# Specify the base image without the --platform flag
FROM pytorch/pytorch:latest

# Install Python packages in a single RUN command to reduce image layers
RUN pip install fire rdkit-pypi gpytorch botorch guacamol wandb selfies \
    pytorch_lightning pandas networkx matplotlib && \
    apt update && \
    apt install -y build-essential libxrender1 libxext6 software-properties-common apt-utils && \
    conda update -n base -c defaults conda && \
    conda install -c conda-forge -y pomegranate && \
    pip install --no-deps molsets fcd-torch

