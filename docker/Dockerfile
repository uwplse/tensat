# TASO parts

FROM nvidia/cuda:10.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.0/cudnn-10.0-linux-x64-v7.6.0.64.tgz && \
    tar -xzf cudnn-10.0-linux-x64-v7.6.0.64.tgz -C /usr/local && \
    rm cudnn-10.0-linux-x64-v7.6.0.64.tgz && \
    ldconfig

RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mv Miniconda3-latest-Linux-x86_64.sh ~/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-latest-Linux-x86_64.sh && \
    ~/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install cmake make
RUN /opt/conda/bin/conda install -c conda-forge protobuf=3.9 numpy onnx
RUN /opt/conda/bin/conda install -c anaconda cython

ENV PATH /opt/conda/bin:$PATH
ENV TASO_HOME /usr/TASO/

#RUN chmod -R a+w /usr/TASO

# Tamago parts
  
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

#RUN git clone https://github.com/uwplse/tamago.git /usr/tamago

#RUN git clone https://github.com/mwillsey/egg.git /usr/egg

RUN mkdir /usr/egg

RUN mkdir /usr/tamago

RUN mkdir /usr/TASO

RUN cd /usr && wget https://www2.graphviz.org/Packages/stable/portable_source/graphviz-2.44.0.tar.gz && tar -xvf graphviz-2.44.0.tar.gz && cd graphviz-2.44.0 && ./configure && make && make install

# gdb parts
#RUN apt-get install -y texinfo
#RUN cd /usr && wget "http://ftp.gnu.org/gnu/gdb/gdb-9.2.tar.gz" && tar -xvzf gdb-9.2.tar.gz && cd gdb-9.2 && mkdir build && cd build && ../configure && make & make install

# or-tools parts
RUN apt-get update && apt-get -y install git wget pkg-config build-essential cmake autoconf libtool zlib1g-dev lsb-release libreadline-dev && apt-get -y install swig
RUN git clone https://github.com/google/or-tools /usr/or-tools

# Install SCIP (MILP solver)
#Get archive https://scip.zib.de/download.php?fname=scipoptsuite-7.0.1.tgz (might need to download to local computer then use scp)
#tar xvf scipoptsuite-7.0.1.tgz
#cd scipoptsuite-7.0.1/
#mkdir build
#cd build
#cmake -DCMAKE_INSTALL_PREFIX=scip/install/dir -DGMP=false -DZIMPL=false -DTPI=tny -DPARASCIP=true ..
#make
#make install

#cd /usr/or-tools
#make third_party
#In Makefile.local, add: UNIX_SCIP_DIR=/usr/scipoptsuite-7.0.1/build/scip/install/dir
#make python
#make test_python
#make install_python



WORKDIR /usr/tamago
