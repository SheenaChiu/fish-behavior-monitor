# ============================================================
# Fish Vitality Monitoring - 推論環境
# ============================================================
# Tested on:
#   OS:     Ubuntu 22.04
#   Python: 3.12
#   CUDA:   13.0 (base image 12.4 + PyTorch cu130 wheel; PyTorch wheel
#                 自帶 CUDA 13 runtime library，base 僅需 driver 相容)
#   GPU:    NVIDIA RTX 5090
#
# Build:
#   docker build -t fish-vitality .
#
# Run inference:
#   docker run --gpus all -v $(pwd):/workspace fish-vitality \
#       python inference.py --video sample.mp4 ...
# ============================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ------------------------------------------------------------
# 1. 系統套件 + Python 3.12
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
        wget \
        curl \
        git \
        software-properties-common \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. 將 python / python3 預設指向 3.12
# ------------------------------------------------------------
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ------------------------------------------------------------
# 3. 安裝 pip 與 setuptools / wheel
#    setuptools 鎖在 70 以下以避免 Triton 相關的 build 錯誤
# ------------------------------------------------------------
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && pip install "setuptools<70.0.0" wheel

# ------------------------------------------------------------
# 4. 安裝 PyTorch (cu130 wheel，與 RTX 5090 相容)
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
        torch==2.12.0 \
        torchvision==0.27.0 \
        torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ------------------------------------------------------------
# 5. 安裝其餘推論依賴
# ------------------------------------------------------------
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 6. 複製專案檔案
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# 7. 預設執行指令 (顯示 inference 的使用說明)
# ------------------------------------------------------------
CMD ["python", "inference.py", "--help"]