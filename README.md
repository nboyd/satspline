# satspline
This repo accompanies the paper [Saturating Splines](https://arxiv.org/abs/1609.06764).
We provide a simple implementation of the algorithm described in the paper to fit generalized
additive models with saturating spline coordinate functions.

### Prerequisites
This project requries both [Rust](https://www.rust-lang.org/en-US/install.html) and
[Julia](https://julialang.org/downloads/).
While they are not required to run the core algorithm, in order to reproduce all of the experiments the Julia libraries listed in REQUIRE must be installed.

First build the Rust components by executing
```
cargo build --release
```
in the root directory of the project.

### Examples
See the examples directory for a few scripts that show how to fit GAMs with our library.
