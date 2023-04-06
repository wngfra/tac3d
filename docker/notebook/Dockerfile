# Copyright (c) 2022 wngfra
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

ARG FROM_IMAGE=nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
FROM $FROM_IMAGE

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install base utilities
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    git \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:${PATH}

# Install Jupyter Lab, and packages for DL and NC
RUN rm ~/miniconda.sh && \
    conda update conda --yes && \
    conda install -c conda-forge --quiet --yes \
    black \
    matplotlib \
    ffmpeg \
    opencv \
    nodejs \
    ipympl \
    isort \
    jupyterlab && \
    conda clean -a -q -y && \
    npm cache clean --force && \
    jupyter lab clean

SHELL ["/bin/bash", "-c", "source ~/.bashrc"]
RUN conda init bash && \
    conda activate base && \
    pip3 install -q -U \
    brian2 \
    mujoco \
    dm_control \
    nengo \
    nengo-gui \
    nengo-extras \
    nengo-dl \
    scipy \
    jupyterlab_code_formatter

RUN pip3 install git+https://github.com/brian-team/brian2tools.git
WORKDIR /workspace

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

EXPOSE 8888
CMD ["jupyter-lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]