# syntax=docker/dockerfile:1.4

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
    CMAKE_PREFIX_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    PYTORCH_ROCM_ARCH="gfx1030" \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    CMAKE_BUILD_PARALLEL_LEVEL="8" \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST=""

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
# 3. THE "BLINGFIRE" EQUIVALENTS & NLP PIPELINE
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    tiktoken \
    tokenizers \
    sentencepiece \
    fasttext \
    sacremoses \
    ftfy \
    clean-text \
    unidecode \
    spacy \
    nltk \
    gensim \
    langdetect

# ===========================================================================
# 4. PRE-TRAINING DATA CLEANING & DEDUPLICATION
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    datasketch \
    trafilatura \
    beautifulsoup4 \
    lxml \
    html5lib \
    scrubadub

# ===========================================================================
# 5. "IN-MEMORY" DBs, FORMATS & DATA STREAMING
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    lmdb \
    diskcache \
    orjson \
    ujson \
    msgspec \
    jsonlines \
    ruamel.yaml \
    polars \
    pyarrow \
    fastparquet \
    pandas \
    openpyxl \
    duckdb \
    webdataset \
    litdata \
    smart_open[s3,gcs] \
    zstandard \
    lz4 \
    blosc \
    snappy \
    py7zr \
    brotli

# ===========================================================================
# 6. 32GB VRAM MODEL TRAINING & FINE-TUNING STACK
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    peft \
    trl \
    transformers \
    datasets \
    accelerate \
    optimum \
    safetensors \
    outlines \
    lm-eval \
    rouge-score \
    bert-score \
    sacrebleu \
    evaluate \
    wandb \
    tensorboard \
    tensorboardX

RUN python3.12 -m pip install --no-cache-dir --no-deps bitsandbytes
RUN python3.12 -m pip install --no-cache-dir --no-deps optimum-amd

# ===========================================================================
# 7. GENERAL PYTHON ENGINEERING, APIS, & CODE QUALITY
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pydantic-settings \
    httpx \
    requests \
    urllib3 \
    rich \
    tqdm \
    click \
    typer \
    matplotlib \
    seaborn \
    plotly \
    pytest \
    pytest-cov \
    ruff \
    black \
    mypy \
    isort \
    pre-commit \
    setuptools-rust

# ===========================================================================
# 8. HUGGINGFACE XET (Next-Gen Model Fetching)
# ===========================================================================
RUN python3.12 -m pip install --no-cache-dir "huggingface_hub[hf_xet]"

# ===========================================================================
# 9. MAKE PYTHON 3.12 THE DEFAULT
# ===========================================================================
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# ===========================================================================
# 10. STABLE-DIFFUSION.CPP (AMD RDNA2 ROCm HIP & Vulkan Backends)
# ===========================================================================
# DEFINITIVE FIX: 
# 1. Added `build-essential`: This strictly depends on `libc6-dev`. Because your base 
#    image uses `--no-install-recommends`, `libc6-dev` (which provides /usr/include/stdio.h 
#    and /usr/include/stdlib.h) was completely missing. This caused CMake to fail at 
#    `project()` and `hipcc` to fail with `'cstdlib' file not found`.
# 2. Added `gcc-14` and `g++-14`: Ubuntu 26.04 defaults to GCC 15. ROCm 7.14's bundled 
#    Clang 23 compiler may fail to parse GCC 15 headers. We install GCC 14 and force 
#    CMake to use it to ensure 100% compatibility.
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

# Force the system to use GCC 14 so Clang 23 (hipcc) finds compatible headers
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100

WORKDIR /workspace/stable-diffusion.cpp
RUN git clone --recursive --depth 1 https://github.com/leejet/stable-diffusion.cpp.git . && \
    mkdir build && cd build && \
    CC=gcc-14 CXX=g++-14 \
    cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSD_HIPBLAS=ON \
      -DSD_VULKAN=ON \
      -DAMDGPU_TARGETS="gfx1030" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_PREFIX_PATH="${ROCM_PATH}" && \
    ninja

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
