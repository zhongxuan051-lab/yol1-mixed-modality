# Supplied DCNv4 sources

Copied from the user's `mianju/ultralytics/nn/extra_modules/DCNv4_op` package.
The source computations and copyright/license headers are retained. Trailing
whitespace and final blank lines were removed, with line endings normalized.
Upstream: <https://github.com/OpenGVLab/DCNv4> (CVPR 2024).
Individual headers identify OpenGVLab MIT code and inherited Apache-2.0 code.

The experiment uses the supplied NCHW wrapper in
`ultralytics/nn/modules/dcnv4.py`, not the upstream token-layout wrapper.
The model's CPU initialization uses an explicit bilinear reference; every GPU
forward/backward uses this extension. No alternative CUDA backend is selected.

Build in the active training environment with `python scripts/install_dcnv4.py`.
Do not copy a compiled binary between different Python/PyTorch/CUDA environments.
