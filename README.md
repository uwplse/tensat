Re-implementation of the [TASO compiler](https://github.com/jiazhihao/TASO)
using [equality saturation](https://mwillsey.com/papers/egg/). Tensat implements
both the graph transformation verifier and the optimizer; the former is complete while
the latter is in progress.

## development

Tensat builds on TASO, so it has the same hardware requirement as TASO. This essentially
means you need GPUs and drivers for [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/)
(if you just want to run the verifier, you don't need GPUs and the regular docker works).
You need the TASO runtime (with its dependencies), rust and
[rust-bindgen](https://github.com/rust-lang/rust-bindgen) to build tensat.

The [`Dockerfile`](docker/Dockerfile) sets this all up for you. Here's the recommended way of
setting up the environment using docker:

- `cd` to `/docker` and execute `docker build --tag tensat:1.0 .` to build the image
- Get the dependent repositories. We need a [`forked version of egg`](https://github.com/yycdavid/egg), and a [`forked version of TASO`](https://github.com/yycdavid/taso). Clone these two repositories to a path of your choice
- Change the `source` parameter of bind mount in run_docker.sh to the path of your choice to tensat, egg, and taso
- Run `./run_docker.sh`
- Now you are inside the docker container, we need to build TASO:
    - Run the following to install:
    ```
    cd /usr/TASO
    mkdir -p build
    cd build
    cmake ..
    sudo make install -j10
    cd /usr/TASO/python
    python setup.py install
    ```
- Then it is good to go

We recommend perusing the
`rust-bindgen` [guide](https://rust-lang.github.io/rust-bindgen/) and related
docs, and note that its c++ support is primitive.

To help debugging, you can install gdb:
```
apt-get update
apt-get install -y texinfo
cd /usr && wget "http://ftp.gnu.org/gnu/gdb/gdb-9.2.tar.gz" && tar -xvzf gdb-9.2.tar.gz && cd gdb-9.2 && mkdir build && cd build && ../configure && make
make install
```

## the verifier
The verifier re-implements TASO's [verify.py](https://github.com/jiazhihao/TASO/blob/master/verify/verify.py).
It takes a list of
transformation rules to be checked and populates an EGraph with the expressions in
these rules. Then it iteratively applies the axioms, checking if all rules are verified
after each round. If so it stops, indicating success; otherwise it continues until the
EGraph saturates. If there are still un-verified rules after saturation, we can
conclude those rules are unsound w.r.t. the axioms. This strategy is faster (~30x in
our simple experiments) than naively
verifying rule-by-rule, because the equality proofs of many rules may overlap, and each
EClass may contain expressions from many different rules.

To run the verifier, uncomment `prove_taso_rules()` in `main.rs/main()`, comment out
`optimize()`, `cd` to project root and execute `cargo run --release taso_rules.txt`.
The `--release` flag turns on rust optimizations.

## the optimizer
The optimizer replaces TASO's [backtracking search](https://cs.stanford.edu/~padon/taso-sosp19.pdf)
with equality saturation. It uses TASO's synthesized rewrite rules. It leverages TASO's
infrastructure for maintaining metadata of the tensor information (like shape), as well as
TASO's cost function that directly executes DL operators.

`run_exp_main.sh` has example commands to run the optimizer. It runs the optimization on TASO's 4 benchmarks and collect various of statistics. `analysis/stats.py` can be used to analyze the statistics and plot results. Uncomment the `-x` flag and argument to save the optimized model into a file. This file can be converted to ONNX format by `TASO/example/load_model.py` (in our fork of TASO).

We support both greedy extraction and ILP extraction. User can control many options through command line flags (see src/main.rs for the flags).
