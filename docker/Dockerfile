FROM tasoml/cuda100
  
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

RUN cd /usr && wget https://www2.graphviz.org/Packages/stable/portable_source/graphviz-2.44.0.tar.gz && tar -xvf graphviz-2.44.0.tar.gz && cd graphviz-2.44.0 && ./configure && make && make install

#RUN apt-get install -y texinfo
#RUN cd /usr && wget "http://ftp.gnu.org/gnu/gdb/gdb-9.2.tar.gz" && tar -xvzf gdb-9.2.tar.gz && cd gdb-9.2 && mkdir build && cd build && ../configure && make & make install

WORKDIR /usr/tamago
