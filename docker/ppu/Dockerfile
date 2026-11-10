# syntax=docker/dockerfile:1

FROM pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001

# The build context must be the repository root so that the source tree of the current working copy is used.
# docker build -f docker/sail/Dockerfile -t deepspeed-sail:local .
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ARG NCCL_SDK_URL=https://pkg.flytiger-eco.com/artifactory/generic-local/NCCL/v2.2.0/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.0.tar.gz

RUN wget -O /tmp/nccl-sdk.tar.gz "${NCCL_SDK_URL}" && \
    tar -xzf /tmp/nccl-sdk.tar.gz -C /usr/local && \
    rm -f /tmp/nccl-sdk.tar.gz

# Load the SDK environment for both build and runtime so that environment variables are not lost after a RUN layer finishes.
COPY <<'EOF' /usr/local/bin/deepspeed-sail-env
#!/bin/bash
set -eo pipefail
# The SDK scripts read undefined variables, so nounset must be enabled only after loading.
source /usr/local/pccl/envsetup.sh
set -u
export NCCL_HOME="${NCCL_HOME:?}"
export NCCL_INCLUDE_DIR="${NCCL_HOME}/include"
export NCCL_LIB_DIR="${NCCL_HOME}/lib"
export CPATH="${NCCL_HOME}/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
EOF

SHELL ["/bin/bash", "/usr/local/bin/deepspeed-sail-env", "/bin/bash", "-eo", "pipefail", "-c"]

# Docker builds typically cannot see devices, so set the backend explicitly to avoid auto-selecting CPU ops.
ENV DS_ACCELERATOR=cuda
# CUDA 13.0 is not covered by the default architecture detection; reuse the architecture list from the PPU workflow.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9"

RUN python -m pip install --no-cache-dir build setuptools wheel ninja && \
    git clone --recurse-submodules https://github.com/deepspeedai/deepspeed-kernels.git /tmp/deepspeed-kernels && \
    cd /tmp/deepspeed-kernels && CUDA_ARCH_LIST="80" CUDA_ARCH_LIST="80" python setup.py bdist_wheel && \
    cp dist/*.whl /opt/ && \
    python -m pip install /opt/*.whl --no-build-isolation -v  && \
    cd /tmp && rm -rf /tmp/deepspeed-kernels

COPY . /opt/deepspeed
WORKDIR /opt/deepspeed

# Disable FP quantizer to stay consistent with the workflow; build and validate the wheel outside the source directory.
# First remove any same-version DeepSpeed that the base image may have preinstalled to ensure the freshly built artifact is installed.
RUN DS_BUILD_OPS=1 DS_BUILD_FP_QUANTIZER=0 \
    python -m build --wheel --no-isolation --outdir /opt/wheels && \
    cd /tmp && \
    python -m pip uninstall -y deepspeed && \
    python -m pip install --no-cache-dir /opt/wheels/deepspeed-*.whl && \
    python -c "import deepspeed; print(deepspeed.__file__, deepspeed.__version__)"

# Keep the default working directory outside the source tree to prevent importing uninstalled source instead of the compiled wheel.
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/usr/local/bin/deepspeed-sail-env"]
CMD ["/bin/bash"]
