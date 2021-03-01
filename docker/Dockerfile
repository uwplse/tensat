# TASO parts

FROM nvidia/cuda:10.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.0/cudnn-10.0-linux-x64-v7.6.0.64.tgz && \
    tar -xzf cudnn-10.0-linux-x64-v7.6.0.64.tgz -C /usr/local && \
    rm cudnn-10.0-linux-x64-v7.6.0.64.tgz && \
    ldconfig

RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    mv Miniconda3-py37_4.8.3-Linux-x86_64.sh ~/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    ~/Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install cmake make
RUN /opt/conda/bin/conda install -c conda-forge protobuf=3.9 numpy onnx
RUN /opt/conda/bin/conda install -c anaconda cython

ENV PATH /opt/conda/bin:$PATH
ENV TASO_HOME /usr/TASO/

# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    llvm-dev libclang-dev clang \
    curl

# Update new packages
RUN apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

ENV LD_LIBRARY_PATH /opt/conda/lib:/usr/local/lib

RUN mkdir /usr/egg

RUN mkdir /usr/tensat

RUN mkdir /usr/TASO

RUN cd /usr && wget https://www2.graphviz.org/Packages/stable/portable_source/graphviz-2.44.0.tar.gz && tar -xvf graphviz-2.44.0.tar.gz && cd graphviz-2.44.0 && ./configure && make && make install

# install or-tools
RUN python -m pip install -U ortools

# vim
RUN apt update
RUN apt install -y vim

WORKDIR /usr/tensat
