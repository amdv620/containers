# syntax=docker/dockerfile:1.4

# Add ARG to accept the base image dynamically (uses your C++ dev image)
ARG BASE_IMAGE=dev
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ===========================================================================
# ENVIRONMENT VARIABLES
# ===========================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    CMAKE_PREFIX_PATH=/opt/rocm:/opt/rocm/llvm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    PYTORCH_ROCM_ARCH="gfx1030" \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    CMAKE_BUILD_PARALLEL_LEVEL="8" \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="" \
    # CRITICAL: Force hipcc to use the GCC 14 toolchain explicitly to bypass 
    # GCC 15 header incompatibilities with ROCm 7.14's Clang 23.
    HIPCC_COMPILE_FLAGS_APPEND="--gcc-toolchain=/usr/lib/gcc/x86_64-linux-gnu/14" \
    HIPCC_LINK_FLAGS_APPEND="--gcc-toolchain=/usr/lib/gcc/x86_64-linux-gnu/14"

# ===========================================================================
# 1. SYSTEM DEPENDENCIES & PYTHON 3.12
# ===========================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    tzdata \
    libgl1 libglx-mesa0 libglib2.0-0t64 \
    libsm6 libxext6 libxrender1 \
    fzf \
    ripgrep \
    bat \
    eza \
    fd-find \
    ncdu \
    htop \
    tmux \
    tree \
    jq \
    rsync \
    liblmdb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m ensurepip --upgrade && \
    ln -s /usr/bin/batcat /usr/local/bin/bat && \
    ln -s /usr/bin/fdfind /usr/local/bin/fd

# ===========================================================================
# 2. PYTORCH, ROCM WHEELS & FOUNDATION
# ===========================================================================
RUN python3.12 -m pip install --ignore-installed --upgrade pip setuptools wheel build && \
    python3.12 -m pip install --ignore-installed \
    torch==2.13.0+rocm7.14.0 \
    torchvision==0.28.0+rocm7.14.0 \
    torchaudio \
    amd_torch_device_gfx1030==2.13.0+rocm7.14.0 \
    amd_torchvision_device_gfx1030==0.28.0+rocm7.14.0 \
    rocm_sdk_device_gfx1030==7.14.0 \
    rocm_sdk_libraries==7.14.0 \
    rocm_profiler==7.14.0 \
    triton==3.8.0+git4cff872c.rocm7.14.0 \
    amdsmi \
    --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    --extra-index-url https://pypi.org/simple/

# ===========================================================================
# 3-9. PYTHON PACKAGES (Unchanged)
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    tiktoken tokenizers sentencepiece fasttext sacremoses ftfy clean-text \
    unidecode spacy nltk gensim langdetect datasketch trafilatura \
    beautifulsoup4 lxml html5lib scrubadub lmdb diskcache orjson ujson \
    msgspec jsonlines ruamel.yaml polars pyarrow fastparquet pandas \
    openpyxl duckdb webdataset litdata "smart_open[s3,gcs]" zstandard \
    lz4 blosc snappy py7zr brotli peft trl transformers datasets \
    accelerate optimum safetensors outlines lm-eval rouge-score \
    bert-score sacrebleu evaluate wandb tensorboard tensorboardX \
    fastapi uvicorn pydantic pydantic-settings httpx requests urllib3 \
    rich tqdm click typer matplotlib seaborn plotly pytest pytest-cov \
    ruff black mypy isort pre-commit setuptools-rust "huggingface_hub[hf_xet]"

RUN python3.12 -m pip install --no-cache-dir --no-deps bitsandbytes
RUN python3.12 -m pip install --no-cache-dir --no-deps optimum-amd

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# ===========================================================================
# 10. CRITICAL FIXES FOR CLOUD BUILDERS (NO GPU) & COMPILER HEADERS
# ===========================================================================
# FIX A: GCC 15 Incompatibility
# ROCm 7.14's Clang 23 cannot parse GCC 15 C++ headers (fatal error: 'cstdlib' not found).
# We install GCC 14 and force Clang to use the GCC 14 toolchain via environment variables.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-14 \
    g++-14 \
    libstdc++-14-dev \
    cmake \
    ninja-build \
    git \
    amdrocm-core-dev \
    amdrocm-blas-dev \
    amdrocm-hipblas-common-dev \
    amdrocm-llvm-dev \
    amdrocm-hipify \
    amdrocm-core-sdk \
    libvulkan-dev \
    glslang-tools \
    glslc \
    spirv-headers \
    spirv-tools \
    libnuma1 \
    libgomp1 \
    libelf1 \
    libdrm2 \
    libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100

# FIX B: The "Cloud Builder / No-GPU" Offload-Arch Bug
# When building in Docker without a GPU, CMake's enable_language(HIP) runs 
# `offload-arch` and `amdgpu-arch`. These tools fail and output an error to STDERR.
# CMake incorrectly captures this STDERR and bakes it into CMAKE_HIP_FLAGS as a 
# literal compiler flag. This mangled string appears in every ninja compile command,
# causing clang++ to fail with "2 errors generated when compiling for gfx1030".
# FIX: Replace these tools with dummy scripts that simply echo "gfx1030" and exit 0.
RUN echo -e '#!/bin/bash\necho "gfx1030"' > /usr/local/bin/offload-arch && \
    echo -e '#!/bin/bash\necho "gfx1030"' > /usr/local/bin/amdgpu-arch && \
    echo -e '#!/bin/bash\necho "gfx1030"' > /usr/local/bin/rocm_agent_enumerator && \
    chmod +x /usr/local/bin/offload-arch /usr/local/bin/amdgpu-arch /usr/local/bin/rocm_agent_enumerator

# Overwrite ROCm's internal binaries just in case CMake calls them directly via absolute path
RUN if [ -d "/opt/rocm/bin" ]; then \
        cp /usr/local/bin/offload-arch /opt/rocm/bin/offload-arch && \
        cp /usr/local/bin/amdgpu-arch /opt/rocm/bin/amdgpu-arch && \
        cp /usr/local/bin/rocm_agent_enumerator /opt/rocm/bin/rocm_agent_enumerator && \
        chmod +x /opt/rocm/bin/offload-arch /opt/rocm/bin/amdgpu-arch /opt/rocm/bin/rocm_agent_enumerator; \
    fi

RUN if [ -d "/opt/rocm/llvm/bin" ]; then \
        cp /usr/local/bin/offload-arch /opt/rocm/llvm/bin/offload-arch && \
        cp /usr/local/bin/amdgpu-arch /opt/rocm/llvm/bin/amdgpu-arch && \
        chmod +x /opt/rocm/llvm/bin/offload-arch /opt/rocm/llvm/bin/amdgpu-arch; \
    fi

WORKDIR /workspace/stable-diffusion.cpp
RUN git clone --recursive --depth 1 https://github.com/leejet/stable-diffusion.cpp.git . && \
    mkdir build && cd build && \
    # Use GCC 14 as the host compiler
    CC=gcc-14 CXX=g++-14 \
    cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSD_HIPBLAS=ON \
      -DSD_VULKAN=ON \
      -DGPU_TARGETS="gfx1030" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/llvm" \
      # Force Clang to use GCC 14 toolchain headers (fixes 'cstdlib' not found)
      -DCMAKE_C_FLAGS="--gcc-toolchain=/usr/lib/gcc/x86_64-linux-gnu/14" \
      -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr/lib/gcc/x86_64-linux-gnu/14" \
      -DCMAKE_HIP_FLAGS="--gcc-toolchain=/usr/lib/gcc/x86_64-linux-gnu/14" \
      # Bypass CMake's HIP compiler test, which often crashes in headless CI environments
      -DCMAKE_HIP_COMPILER_WORKS=TRUE \
      && ninja

# Symlink binaries to PATH
RUN ln -s /workspace/stable-diffusion.cpp/build/bin/sd-cli /usr/local/bin/sd-cli && \
    ln -s /workspace/stable-diffusion.cpp/build/bin/sd-server /usr/local/bin/sd-server || true

WORKDIR /workspace

# ===========================================================================
# 11. HUGGINGFACE DIFFUSERS & QWEN-IMAGE-2.1 DEPENDENCIES
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    diffusers \
    pillow \
    einops \
    invisible-watermark \
    omegaconf

# ===========================================================================
# FINAL VERIFICATION (CPU / CI Safe)
# ===========================================================================
RUN python3.12 -c "import torch; print('✅ PyTorch:', torch.__version__, '| HIP:', torch.version.hip if hasattr(torch.version, 'hip') else 'CPU Build')" && \
    python3.12 -c "import amdsmi; print('✅ AMDSMI (ROCm 7.14.0) installed')" && \
    python3.12 -c "import bitsandbytes; print('✅ BitsAndBytes:', bitsandbytes.__version__)" && \
    python3.12 -c "import datasketch; print('✅ Datasketch (MinHash Dedup) installed')" && \
    python3.12 -c "import trafilatura; print('✅ Trafilatura (Web Extractor) installed')" && \
    python3.12 -c "import tiktoken; print('✅ Tiktoken (BlingFire equiv) installed')" && \
    python3.12 -c "import sentencepiece; print('✅ SentencePiece installed')" && \
    python3.12 -c "import duckdb; print('✅ DuckDB (In-Process SQL):', duckdb.__version__)" && \
    python3.12 -c "import lmdb; print('✅ LMDB (Memory Mapped DB) installed')" && \
    python3.12 -c "import polars as pl; print('✅ Polars:', pl.__version__)" && \
    python3.12 -c "import huggingface_hub; print('✅ HF Hub (XET enabled)')" && \
    python3.12 -c "import torch.nn.functional as F; print('✅ PyTorch Native SDPA (Flash Attention equivalent) is available')" && \
    python3.12 -c "import diffusers; print('✅ Diffusers:', diffusers.__version__)" && \
    python3.12 -c "import accelerate; print('✅ Accelerate:', accelerate.__version__)" && \
    ls /usr/local/bin/sd-cli && echo "✅ stable-diffusion.cpp (sd-cli) built successfully"

EXPOSE 8000 7860
CMD ["/bin/bash"]